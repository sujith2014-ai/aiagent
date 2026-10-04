"""Selective learning decision engine. Strategies are tried from cheapest to most expensive; a candidate must pass
the gates (new-domain verification from the environment, old-domain regression) before it is promoted.
Order: metadata_update (no neural change) -> adapt_existing (with replay) -> new_module."""
from __future__ import annotations
from training.learning_package.spec import TaskSpec, spec_to_learning_package, sample_spec
from training.service import BuildService, PROMOTE_MIN_TEST_ACC


class SelectiveLearner:
    def __init__(self, svc: BuildService, cli):
        self.svc, self.cli = svc, cli
        self.history: list[dict] = []

    def _next_version(self, cap_id: str) -> str:
        cur = next(c for c in self.cli.list() if c["capability_id"] == cap_id)["active_version"]
        major, minor, patch = map(int, cur.split("."))
        return f"{major}.{minor}.{patch + 1}"

    def _import(self, path):
        return self.cli.import_caps(str(path))[0]

    def extend_domain(self, cap_id: str, spec: TaskSpec, env_examples, old_eval, new_cap_keywords=None, allow=("metadata_update", "adapt_existing", "new_module"),
                      teacher_name="teacher") -> dict:
        """The capability must now also work on `spec`'s domain. Returns the decision log."""
        log = {"capability_id": cap_id, "attempts": [], "chosen": None}
        new_lp = spec_to_learning_package(spec, teacher_name)
        old_acc = self.svc.eval_existing(cap_id, *old_eval)
        # 1. no neural change: does the existing module already solve the new domain?
        if "metadata_update" in allow:
            acc = self.svc.eval_existing(cap_id, *env_examples)
            a = {"strategy": "metadata_update", "new_acc_verification": acc, "passed": acc >= PROMOTE_MIN_TEST_ACC}
            log["attempts"].append(a)
            if a["passed"]:
                import numpy as np
                st = self.svc.models[cap_id]["stats"]; ns = {"min": np.min(new_lp.train_x, 0), "max": np.max(new_lp.train_x, 0)}
                stats = {**st, "min": np.minimum(st["min"], ns["min"]).tolist(), "max": np.maximum(st["max"], ns["max"]).tolist()}
                b = self.svc.repackage(cap_id, self._next_version(cap_id), stats=stats, strategy="metadata_update", note="domain already covered")
                imp = self._import(b.cap_path)
                a["activated"] = imp["activated"]
                if imp["activated"]:
                    log["chosen"] = "metadata_update"; self.history.append(log); return log
        # 2. adapt the existing module (with replay of old training data)
        if "adapt_existing" in allow:
            b = self.svc.adapt(cap_id, new_lp, env_examples, old_eval, self._next_version(cap_id), replay=True)
            a = {**b.report, "passed": b.promoted, "reason": b.reason}
            log["attempts"].append(a)
            if b.promoted:
                imp = self._import(b.cap_path)
                a["activated"] = imp["activated"]
                if imp["activated"]:
                    self.svc.commit(); log["chosen"] = "adapt_existing"; self.history.append(log); return log
        # 3. new module for the new domain/rule under a new capability id
        if "new_module" in allow:
            new_id = f"{cap_id}__ext"
            spec2 = TaskSpec.from_dict({**spec.__dict__, "capability_id": new_id, "keywords": new_cap_keywords or spec.keywords})
            lp2 = spec_to_learning_package(spec2, teacher_name)
            used = set(map(tuple, lp2.train_x)) | set(map(tuple, lp2.validation_x))
            held = sample_spec(spec2, 400 if not all("grid" in d for d in spec2.domain) else 80, 7, exclude=used, unique=True)
            b = self.svc.build(lp2, "0.1.0", {c["capability_id"] for c in self.cli.list()}, heldout=held, verification=env_examples)
            a = {"strategy": "new_module", "passed": b.promoted, "reason": b.reason, "params_added": b.report["attempts"][-1]["params"] if b.report.get("attempts") else None,
                 "new_acc_verification": b.report.get("verification_accuracy")}
            log["attempts"].append(a)
            if b.promoted:
                imp = self._import(b.cap_path)
                a["activated"] = imp["activated"]
                if imp["activated"]:
                    log["chosen"] = "new_module"; log["new_capability_id"] = new_id; self.history.append(log); return log
        self.history.append(log)
        return log
