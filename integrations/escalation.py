"""Escalation loop: local solve -> (NEEDS_HELP) -> minimal-context teacher request -> validated response ->
selective action -> retry. Honest about every failure mode; offline requests are queued, never faked."""
from __future__ import annotations
import json, time
from pathlib import Path
from integrations.teacher.provider import TeacherProvider, make_help_request
from integrations.teacher.response import parse_response, InvalidTeacherResponse
from training.learning_package.schema import validate, InvalidLearningPackage
from training.information import classify_information


class Escalator:
    def __init__(self, cli, teacher: TeacherProvider, service, workdir: Path, online=True, max_teacher_calls_per_task=1):
        self.cli, self.teacher, self.service = cli, teacher, service
        self.workdir = Path(workdir); self.workdir.mkdir(parents=True, exist_ok=True)
        self.online, self.max_calls = online, max_teacher_calls_per_task
        self.known: set[str] = {c["capability_id"] for c in cli.list()}
        self.memory: list[dict] = []
        self.log: list[dict] = []
        self.teacher_calls = 0
        self.versions: dict[str, int] = {}

    def _queue(self, entry):
        with open(self.workdir / "pending_help.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    def solve(self, intent: str, x: list[float]) -> dict:
        r = self.cli.solve(intent, x)
        rec = {"intent": intent, "path": ["local"], "teacher_called": False}
        if r["result"] == "ANSWER":
            rec.update(result="ANSWER", status=r["status"], capability=r["capability_id"], label=r["label"], flags=r.get("flags", []))
            self.log.append(rec); return rec
        # NEEDS_HELP
        req = make_help_request(intent, len(x), list(self.known), {"reason_code": r.get("reason_code")}, r.get("reason", ""))
        if not self.online:
            self._queue({"ts": time.time(), "request": json.loads(req.to_json())})
            rec.update(result="QUEUED_OFFLINE", path=["local", "queued"]); self.log.append(rec); return rec
        self.teacher_calls += 1; rec["teacher_called"] = True; rec["path"].append("teacher")
        try:
            resp = parse_response(self.teacher.advise(req), self.known)
        except InvalidTeacherResponse as e:
            rec.update(result="NEEDS_HELP", reason=f"teacher response rejected: {e}"); self.log.append(rec); return rec
        rec["teacher_action"] = resp.action
        if resp.action == "new_capability":
            lp = resp.learning_package
            if classify_information(lp.description) == "MEMORY" or lp.info_kind != "CAPABILITY":
                rec.update(result="NEEDS_HELP", reason="teacher proposed neural training for non-capability information"); self.log.append(rec); return rec
            try:
                validate(lp, self.known)
            except InvalidLearningPackage as e:
                rec.update(result="NEEDS_HELP", reason=f"learning package rejected: {e}"); self.log.append(rec); return rec
            ver = self.versions.get(lp.capability_id, 0) + 1
            built = self.service.build(lp, f"0.{ver}.0", self.known)
            if not built.promoted:
                rec.update(result="NEEDS_HELP", reason=built.reason); self.log.append(rec); return rec
            imp = self.cli.import_caps(str(built.cap_path))[0]
            if not imp["activated"]:
                rec.update(result="NEEDS_HELP", reason=f"import failed at {imp['failed_step']}"); self.log.append(rec); return rec
            self.versions[lp.capability_id] = ver; self.known.add(lp.capability_id); rec["path"].append("learn")
            again = self.cli.solve(intent, x)
            if again["result"] == "ANSWER":
                rec.update(result="ANSWER", status=again["status"], capability=again["capability_id"], label=again["label"], learned=lp.capability_id)
            else:
                rec.update(result="NEEDS_HELP", reason="still unresolved after learning")
        elif resp.action == "use_memory":
            self.memory.append({"text": resp.memory_text, "ts": time.time()})
            rec.update(result="MEMORY_UPDATED", path=rec["path"] + ["memory"])
        elif resp.action == "reroute":
            again = self.cli.solve(resp.reroute_intent, x)
            rec.update(result=again["result"], path=rec["path"] + ["reroute"], capability=again.get("capability_id"), label=again.get("label"))
        elif resp.action in ("external_research", "request_tool"):
            # OpenClaw is not integrated before Phase 8: report honestly, do not pretend research happened
            rec.update(result="NEEDS_EXTERNAL", needs=resp.action, detail=resp.research_query or resp.tool)
        else:
            rec.update(result="NEEDS_HELP", reason="teacher cannot help")
        self.log.append(rec)
        return rec
