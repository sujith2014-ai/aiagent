"""The tool-growth gate: schema -> compile -> static -> sandbox run -> bundled tests -> independent spec cases -> properties -> routing conflicts -> operator approval.
Nothing here can approve: approval is an Ed25519-free, hash-bound record created only by the operator-only `aicli approve` command and checked by the Rust policy engine."""
from __future__ import annotations
import json, time
from dataclasses import dataclass, field
from integrations.toolgrowth.package import Candidate, request_hash_of
from integrations.toolgrowth.static import analyze
from integrations.toolgrowth.sandbox import SandboxConfig, run_batch

STAGES = ["schema", "compile", "static", "sandbox_run", "tests", "spec", "properties", "routing", "approval"]


@dataclass
class Report:
    tool_id: str
    version: str
    code_sha256: str
    stages: list = field(default_factory=list)
    status: str = "REJECTED"            # REJECTED | AWAITING_APPROVAL | APPROVED
    failed_stage: str | None = None
    approval: dict | None = None
    request_hash: str | None = None

    def public(self) -> dict:
        return {"tool_id": self.tool_id, "version": self.version, "code_sha256": self.code_sha256, "status": self.status, "failed_stage": self.failed_stage,
                "stages": self.stages, "approval": self.approval}

    def feedback(self) -> dict:
        """What the generator is allowed to learn about a failure: the stage and a generic message. Never expected outputs of independent spec cases."""
        f = next((s for s in self.stages if not s["ok"]), None)
        return {"failed_stage": self.failed_stage, "message": (f["detail"] if f and self.failed_stage != "spec" else "independent spec cases failed") if f else None}


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


class Pipeline:
    def __init__(self, broker, reserved_keyword_sets: list | None = None, cfg: SandboxConfig | None = None, static: bool = True):
        self.broker, self.reserved, self.cfg, self.static = broker, [set(k) for k in (reserved_keyword_sets or [])], cfg or SandboxConfig(), static

    def evaluate(self, cand: Candidate, spec_cases: list, task_id: str = "tool-growth", *, static: bool | None = None) -> Report:
        static = self.static if static is None else static
        rep = Report(cand.tool_id, cand.version, cand.code_sha256())

        def stage(name, ok, detail="", t0=None, **extra):
            rep.stages.append({"stage": name, "ok": ok, "detail": detail, "seconds": round(time.perf_counter() - t0, 4) if t0 else 0.0, **extra})
            if not ok: rep.failed_stage = name
            return ok

        t0 = time.perf_counter(); problems = cand.schema_problems()
        if not stage("schema", not problems, "; ".join(problems), t0): return rep
        t0 = time.perf_counter()
        try: compile(cand.source, "tool.py", "exec"); ok, d = True, ""
        except SyntaxError as e: ok, d = False, f"line {e.lineno}: {e.msg}"
        if not stage("compile", ok, d, t0): return rep
        t0 = time.perf_counter()
        if static:
            a = analyze(cand.source, cand.permissions)
            if not stage("static", a["ok"], "; ".join(f"line {f['line']}: {f['message']}" for f in a["findings"][:4]), t0, findings=a["findings"], derived_permissions=a["derived_permissions"]): return rep
        else:
            stage("static", True, "skipped (ablation)", t0)
        cases = [t["input"] for t in cand.tests] + [t["input"] for t in spec_cases]
        t0 = time.perf_counter(); run1 = run_batch(cand.source, cases + cases, self.cfg)
        if not stage("sandbox_run", run1.ok, f"{run1.kind}: {run1.detail}" if not run1.ok else "", t0, kind=run1.kind, network_isolated=run1.network_isolated): return rep
        n = len(cases); first, second = run1.results[:n], run1.results[n:]
        nb = len(cand.tests)
        bad = [i for i, t in enumerate(cand.tests) if not (first[i]["ok"] and _same(first[i]["output"], t["expected"]))]
        if not stage("tests", not bad, f"{len(bad)} of {nb} bundled tests failed (first: #{bad[0]})" if bad else "", None, passed=nb - len(bad), total=nb): return rep
        sbad = [i for i, t in enumerate(spec_cases) if not (first[nb + i]["ok"] and _same(first[nb + i]["output"], t["expected"]))]
        if not stage("spec", not sbad, f"{len(sbad)} of {len(spec_cases)} independent spec cases failed", None, passed=len(spec_cases) - len(sbad), total=len(spec_cases)): return rep
        t0 = time.perf_counter(); problems = []
        if not all(_same(a, b) for a, b in zip(first, second)): problems.append("results differ when the same inputs are run again in one process (hidden state)")
        run2 = run_batch(cand.source, cases, self.cfg)
        if not run2.ok or not all(_same(a, b) for a, b in zip(first, run2.results)): problems.append("results differ across fresh processes (non-deterministic)")
        if not all(isinstance(r.get("output"), dict) for r in first if r["ok"]): problems.append("outputs must be JSON objects")
        if not stage("properties", not problems, "; ".join(problems), t0): return rep
        t0 = time.perf_counter(); kws = {k.lower() for k in cand.keywords}
        clash = next((sorted(kws & r) for r in self.reserved if len(kws & r) >= 2), None)
        if not stage("routing", not clash and len(kws) >= 2, f"keywords overlap an existing capability's routing keywords {clash}" if clash else ("at least 2 keywords are required" if len(kws) < 2 else ""), t0): return rep
        t0 = time.perf_counter(); params = cand.approval_params(); dec = self.broker.check(task_id, "tool.install", params); rep.approval = dec
        rep.request_hash = request_hash_of(params)
        self.broker.audit(task=task_id, action="tool.install", params_sha256=rep.request_hash, params_preview={k: str(v)[:60] for k, v in params.items()}, decision=dec["effect"], rule=dec["rule_id"], reason=dec["reason"])
        if dec["effect"] == "deny": stage("approval", False, f"denied by policy: {dec['reason']}", t0); return rep
        rep.status = "APPROVED" if dec["effect"] == "allow" else "AWAITING_APPROVAL"
        stage("approval", True, "approved by operator" if rep.status == "APPROVED" else f"awaiting operator approval of request {rep.request_hash}", t0)
        return rep
