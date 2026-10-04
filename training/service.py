"""Build/validation service: the only component that trains candidates and holds the signing key.
Candidates never replace active capabilities until all gates pass."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from training.learning_package.schema import LearningPackage, validate
import numpy as np, torch
from training.trainer.train import train_candidate, accuracy
from training.exporters.onnx_export import export_onnx, verify_export
from packages.capbuild import build_cap, load_private

PROMOTE_MIN_TEST_ACC = 0.95


def _logits(model, x):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy().astype(np.float64)


def _softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == np.asarray(y)).astype(float)
    edges = np.linspace(0, 1, bins + 1); tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(tot)


def fit_temperature(logits, y):
    """Temperature scaling on validation data: grid search minimizing NLL."""
    y = np.asarray(y); best, bt = 1e18, 1.0
    for t in np.geomspace(0.05, 4.0, 90):
        p = _softmax(logits / t)
        nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()
        if nll < best:
            best, bt = nll, float(t)
    return bt


def input_stats(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "mean": a.mean(0).tolist(), "std": a.std(0).tolist()}


@dataclass
class BuildResult:
    promoted: bool
    reason: str
    cap_path: Path | None = None
    report: dict = field(default_factory=dict)


class BuildService:
    def __init__(self, keydir: Path, signer_id: str, workdir: Path, oracle):
        self.key = load_private(signer_id, keydir)
        self.signer_id, self.workdir, self.oracle = signer_id, workdir, oracle

    def build(self, lp: LearningPackage, version: str, known_caps: set[str], attempts=((32, 32), (64, 64))) -> BuildResult:
        validate(lp, known_caps)                                   # 1. LearningPackage validation
        used = set(map(tuple, lp.train_x)) | set(map(tuple, lp.validation_x))
        tx, ty = self.oracle.heldout(lp.capability_id, used)       # 2. independent generalization set
        rep = {"capability_id": lp.capability_id, "attempts": []}
        for i, hidden in enumerate(attempts):                      # 3. isolated training of a *new* module
            model, tr = train_candidate(lp, hidden=hidden, seed=i)
            test_acc = accuracy(model, tx, ty)
            rep["attempts"].append({**tr, "heldout_accuracy": test_acc, "heldout_n": len(tx)})
            if test_acc >= PROMOTE_MIN_TEST_ACC:
                break
        else:
            return BuildResult(False, f"candidate rejected: heldout accuracy {test_acc:.3f} < {PROMOTE_MIN_TEST_ACC}", None, rep)
        onnx_bytes = export_onnx(model, lp.input_dim)              # 4. export + parity check vs PyTorch
        rep["onnx_max_abs_diff"] = verify_export(model, onnx_bytes, tx)
        T = fit_temperature(_logits(model, lp.validation_x), lp.validation_y)       # fitted on validation, reported on held-out
        lg = _logits(model, tx)
        rep["calibration"] = {"temperature": T, "ece_heldout_before": ece(_softmax(lg), ty), "ece_heldout_after": ece(_softmax(lg / T), ty)}
        tests = list(zip(tx[:300], ty[:300]))
        out = self.workdir / f"{lp.capability_id}-{version}.cap"
        manifest = build_cap(out, capability_id=lp.capability_id, version=version, model_bytes=onnx_bytes,
            params=tr["params"], input_dim=lp.input_dim, labels=lp.labels, tests=tests, keywords=lp.keywords,
            description=lp.description, signer_id=self.signer_id, signer_key=self.key, device_caps=lp.device_caps,
            dependencies=lp.existing_dependencies, input_stats=input_stats(lp.train_x + lp.edge_cases_x),
            calibration={"method": "temperature", "temperature": T}, min_accuracy=PROMOTE_MIN_TEST_ACC - 0.05,
            provenance={"learning_package": {k: lp.provenance.get(k) for k in ("source", "teacher", "created", "verified")},
                        "training": {k: tr[k] for k in ("epochs", "hidden", "seed", "train_seconds", "val_accuracy")},
                        "heldout_accuracy": test_acc, "strategy": lp.strategy})
        rep["package_id"] = manifest["package_id"]
        return BuildResult(True, "promoted", out, rep)
