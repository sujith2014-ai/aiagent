"""Build/validation service: the only component that trains candidates and holds the signing key.
Candidates never replace active capabilities until all gates pass."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from training.learning_package.schema import LearningPackage, validate
import copy, numpy as np, torch, torch.nn as nn
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
        self.models: dict[str, dict] = {}   # capability_id -> trained state kept for adaptation / metadata updates

    def build(self, lp: LearningPackage, version: str, known_caps: set[str], attempts=((32, 32), (64, 64), (64, 64)),
              heldout=None, verification=None, min_verification=PROMOTE_MIN_TEST_ACC) -> BuildResult:
        validate(lp, known_caps)                                   # 1. LearningPackage validation
        used = set(map(tuple, lp.train_x)) | set(map(tuple, lp.validation_x))
        tx, ty = heldout if heldout is not None else self.oracle.heldout(lp.capability_id, used)   # 2. generalization set
        rep = {"capability_id": lp.capability_id, "attempts": []}
        for i, hidden in enumerate(attempts):                      # 3. isolated training of a *new* module
            model, tr = train_candidate(lp, hidden=hidden, seed=i, epochs=400 + 200 * (i // 2))
            test_acc = accuracy(model, tx, ty)
            att = {**tr, "heldout_accuracy": test_acc, "heldout_n": len(tx)}
            rep["attempts"].append(att)
            vacc = None
            if verification is not None:                           # environment-supplied examples (independent of the teacher)
                vacc = accuracy(model, verification[0], verification[1])
                att["verification_accuracy"] = vacc
                rep["verification_accuracy"] = vacc; rep["verification_n"] = len(verification[0])
            if test_acc >= PROMOTE_MIN_TEST_ACC and (vacc is None or vacc >= min_verification):
                break
        else:
            why = f"heldout accuracy {test_acc:.3f} < {PROMOTE_MIN_TEST_ACC}" if test_acc < PROMOTE_MIN_TEST_ACC else \
                  f"environment verification accuracy {vacc:.3f} < {min_verification}"
            return BuildResult(False, f"candidate rejected after {len(attempts)} attempts: {why}", None, rep)
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
                        "heldout_accuracy": test_acc, "strategy": lp.strategy,
                        "verification_accuracy": rep.get("verification_accuracy"), "verified_by_environment": verification is not None})
        rep["package_id"] = manifest["package_id"]
        self.models[lp.capability_id] = {"model": model, "hidden": list(tr["hidden"]), "version": version, "onnx": onnx_bytes, "tests": tests,
            "replay": (lp.train_x[:600], lp.train_y[:600]), "val": (lp.validation_x, lp.validation_y), "heldout": (tx, ty),
            "stats": input_stats(lp.train_x + lp.edge_cases_x), "labels": lp.labels, "keywords": list(lp.keywords),
            "description": lp.description, "temperature": T, "params": tr["params"], "input_dim": lp.input_dim,
            "provenance": {"base": lp.provenance.get("source")}}
        return BuildResult(True, "promoted", out, rep)

    # ---- strategies that do not train a new module -------------------------------------------------------------
    def eval_existing(self, cap_id: str, xs, ys) -> float:
        return accuracy(self.models[cap_id]["model"], xs, ys)

    def repackage(self, cap_id: str, version: str, extra_keywords=(), stats=None, strategy="router_update", note="") -> BuildResult:
        """Same model bytes, new signed manifest (routing hints and/or widened input stats). No neural change."""
        m = self.models[cap_id]
        kws = list(dict.fromkeys(m["keywords"] + list(extra_keywords)))
        out = self.workdir / f"{cap_id}-{version}.cap"
        manifest = build_cap(out, capability_id=cap_id, version=version, model_bytes=m["onnx"], params=m["params"], input_dim=m["input_dim"],
            labels=m["labels"], tests=m["tests"], keywords=kws, description=m["description"], signer_id=self.signer_id, signer_key=self.key,
            min_accuracy=PROMOTE_MIN_TEST_ACC - 0.05, input_stats=stats or m["stats"], calibration={"method": "temperature", "temperature": m["temperature"]},
            provenance={**m["provenance"], "strategy": strategy, "base_version": m["version"], "note": note})
        m["keywords"], m["version"] = kws, version
        if stats:
            m["stats"] = stats
        return BuildResult(True, strategy, out, {"package_id": manifest["package_id"], "strategy": strategy})

    def adapt(self, cap_id: str, new_lp: LearningPackage, verification, old_eval, version: str, replay=True, build=True,
              epochs=200, lr=2e-3, min_old_acc_drop=0.01) -> BuildResult:
        """Fine-tune the existing module on a new/extended domain. With replay=True old training examples are mixed in."""
        st = self.models[cap_id]
        model = copy.deepcopy(st["model"])
        old_before = accuracy(model, *old_eval)
        x = new_lp.train_x + new_lp.edge_cases_x; y = new_lp.train_y + new_lp.edge_cases_y
        vx, vy = list(new_lp.validation_x), list(new_lp.validation_y)
        if replay:
            x = x + st["replay"][0]; y = y + st["replay"][1]
            vx = vx + st["val"][0]; vy = vy + st["val"][1]
        xt, yt = torch.tensor(x, dtype=torch.float32), torch.tensor(y)
        opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = nn.CrossEntropyLoss()
        best, best_state, bad = -1.0, None, 0
        torch.manual_seed(0)
        for ep in range(epochs):
            model.train(); perm = torch.randperm(len(x))
            for i in range(0, len(x), 128):
                idx = perm[i:i + 128]; opt.zero_grad(); lossf(model(xt[idx]), yt[idx]).backward(); opt.step()
            va = accuracy(model, vx, vy)
            if va > best + 1e-9:
                best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= 40:
                    break
        model.load_state_dict(best_state)
        rep = {"strategy": "adapt_existing" if replay else "adapt_existing_no_replay", "replay": replay, "epochs": ep + 1, "old_acc_before": old_before,
               "old_acc_after": accuracy(model, *old_eval), "new_acc_verification": accuracy(model, *verification), "params_added": 0}
        if rep["new_acc_verification"] < PROMOTE_MIN_TEST_ACC:
            return BuildResult(False, f"new-domain verification {rep['new_acc_verification']:.3f} < {PROMOTE_MIN_TEST_ACC}", None, rep)
        if rep["old_acc_after"] < old_before - min_old_acc_drop:
            return BuildResult(False, f"regression: old-domain accuracy {old_before:.3f} -> {rep['old_acc_after']:.3f}", None, rep)
        if not build:
            return BuildResult(True, "dry-run passed", None, rep)
        onnx_bytes = export_onnx(model, new_lp.input_dim)
        old_x = np.asarray(st["stats"]["min"]), np.asarray(st["stats"]["max"])
        ns = input_stats(new_lp.train_x + new_lp.edge_cases_x)
        stats = {"min": np.minimum(old_x[0], ns["min"]).tolist(), "max": np.maximum(old_x[1], ns["max"]).tolist(),
                 "mean": ((np.asarray(st["stats"]["mean"]) + ns["mean"]) / 2).tolist(), "std": np.maximum(st["stats"]["std"], ns["std"]).tolist()}
        T = fit_temperature(_logits(model, vx), vy)
        tests = list(zip(new_lp.validation_x[:150], new_lp.validation_y[:150])) + list(zip(old_eval[0][:150], old_eval[1][:150]))
        out = self.workdir / f"{cap_id}-{version}.cap"
        manifest = build_cap(out, capability_id=cap_id, version=version, model_bytes=onnx_bytes, params=st["params"], input_dim=new_lp.input_dim,
            labels=st["labels"], tests=tests, keywords=st["keywords"], description=st["description"], signer_id=self.signer_id, signer_key=self.key,
            min_accuracy=PROMOTE_MIN_TEST_ACC - 0.05, input_stats=stats, calibration={"method": "temperature", "temperature": T},
            provenance={**st["provenance"], "strategy": "adapt_existing", "base_version": st["version"], "replay": replay,
                        "old_acc_before": old_before, "old_acc_after": rep["old_acc_after"], "new_acc_verification": rep["new_acc_verification"]})
        rep["package_id"] = manifest["package_id"]
        self._pending = {"cap_id": cap_id, "update": {"model": model, "version": version, "onnx": onnx_bytes, "tests": tests, "stats": stats,
                         "temperature": T, "replay": (st["replay"][0] + new_lp.train_x[:300], st["replay"][1] + new_lp.train_y[:300]),
                         "val": (vx, vy), "heldout": st["heldout"]}}
        return BuildResult(True, "promoted", out, rep)

    def commit(self):
        """Call after the runtime has activated the candidate built by `adapt`."""
        p = self._pending
        self.models[p["cap_id"]].update(p["update"])
        self._pending = None

