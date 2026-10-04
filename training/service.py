"""Build/validation service: the only component that trains candidates and holds the signing key.
Candidates never replace active capabilities until all gates pass."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from training.learning_package.schema import LearningPackage, validate
from training.trainer.train import train_candidate, accuracy
from training.exporters.onnx_export import export_onnx, verify_export
from packages.capbuild import build_cap, load_private

PROMOTE_MIN_TEST_ACC = 0.95


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
        tests = list(zip(tx[:300], ty[:300]))
        out = self.workdir / f"{lp.capability_id}-{version}.cap"
        manifest = build_cap(out, capability_id=lp.capability_id, version=version, model_bytes=onnx_bytes,
            params=tr["params"], input_dim=lp.input_dim, labels=lp.labels, tests=tests, keywords=lp.keywords,
            description=lp.description, signer_id=self.signer_id, signer_key=self.key, device_caps=lp.device_caps,
            dependencies=lp.existing_dependencies, min_accuracy=PROMOTE_MIN_TEST_ACC - 0.05,
            provenance={"learning_package": {k: lp.provenance.get(k) for k in ("source", "teacher", "created", "verified")},
                        "training": {k: tr[k] for k in ("epochs", "hidden", "seed", "train_seconds", "val_accuracy")},
                        "heldout_accuracy": test_acc, "strategy": lp.strategy})
        rep["package_id"] = manifest["package_id"]
        return BuildResult(True, "promoted", out, rep)
