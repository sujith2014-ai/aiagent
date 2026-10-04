"""The growth loop: intent -> installed tool? -> if not: NEEDS_TOOL -> generate -> gates -> (operator approval, injected, never created here) -> install -> call."""
from __future__ import annotations
import time
from integrations.toolgrowth.generator import ToolRequest, ToolGenerator, GeneratorUnavailable


class GrowthLoop:
    def __init__(self, pipeline, registry, host, index, generator: ToolGenerator, max_attempts: int = 3):
        self.pipeline, self.registry, self.host, self.index, self.generator, self.max_attempts = pipeline, registry, host, index, generator, max_attempts

    def handle(self, request: ToolRequest, spec_cases: list, inp: dict, operator=None) -> dict:
        """`operator(candidate, report) -> bool` is the human. It is a parameter on purpose: this class cannot approve anything itself."""
        trace = {"intent": request.intent, "attempts": []}
        t0 = time.perf_counter()
        hit = self.index.match(request.intent)
        if hit:
            return {**self.host.call(hit, inp, task_id=request.task_id), "path": "existing_tool", "trace": trace}
        trace["gap"] = "NEEDS_TOOL"
        fb = None
        for attempt in range(self.max_attempts):
            try: cand = self.generator.generate(request, attempt, fb)
            except GeneratorUnavailable as e: return {"ok": False, "refused": "GENERATOR_UNAVAILABLE", "reason": str(e), "path": "gap", "trace": trace}
            if cand is None: break
            rep = self.pipeline.evaluate(cand, spec_cases, task_id=request.task_id)
            trace["attempts"].append({"attempt": attempt, "version": cand.version, "status": rep.status, "failed_stage": rep.failed_stage, "feedback_given": rep.feedback() if rep.status == "REJECTED" else None})
            if rep.status == "REJECTED": fb = rep.feedback(); continue
            if rep.status == "AWAITING_APPROVAL":
                if operator is None or not operator(cand, rep): return {"ok": False, "refused": "AWAITING_OPERATOR", "reason": f"operator approval required for request {rep.request_hash}", "path": "gap", "trace": trace}
            res = self.registry.install(cand, spec_cases, task_id=request.task_id)
            trace["attempts"][-1]["installed"] = res["installed"]
            if not res["installed"]: return {"ok": False, "refused": "INSTALL_FAILED", "reason": res["reason"], "path": "gap", "trace": trace}
            trace["growth_seconds"] = round(time.perf_counter() - t0, 3)
            return {**self.host.call(cand.tool_id, inp, task_id=request.task_id), "path": "grown", "trace": trace}
        return {"ok": False, "refused": "NO_ACCEPTABLE_CANDIDATE", "reason": "all attempts rejected by the gates", "path": "gap", "trace": trace}
