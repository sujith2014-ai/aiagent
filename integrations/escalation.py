"""Escalation loop: local solve -> (NEEDS_HELP) -> minimal-context teacher request -> validated response ->
selective action -> retry. Honest about every failure mode; offline requests are queued, never faked."""
from __future__ import annotations
import json, time
from pathlib import Path
from integrations.teacher.provider import TeacherProvider, make_help_request
from integrations.teacher.response import parse_response, InvalidTeacherResponse
from integrations.teacher.http_providers import TeacherUnavailable
from training.learning_package.spec import TaskSpec, InvalidSpec, spec_to_learning_package, sample_spec
from training.safe_expr import UnsafeExpression
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
        self.routing_suite: list[tuple[str, list[float], str]] = []   # (intent, input, capability) served KNOWN: regression suite for routing changes
        self.router_updates = 0
        self.persist_router_updates = True

    def _queue(self, entry):
        with open(self.workdir / "pending_help.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    def solve(self, intent: str, x: list[float], examples: tuple | None = None) -> dict:
        """`examples`: optional (xs, ys) labelled examples supplied by the environment/user, used to verify teacher rules."""
        r = self.cli.solve(intent, x)
        rec = {"intent": intent, "path": ["local"], "teacher_called": False}
        if r["result"] == "ANSWER":
            rec.update(result="ANSWER", status=r["status"], capability=r["capability_id"], label=r["label"], flags=r.get("flags", []))
            if r["status"] == "KNOWN" and len(self.routing_suite) < 200:
                self.routing_suite.append((intent, x, r["capability_id"]))
            self.log.append(rec); return rec
        # NEEDS_HELP
        req = make_help_request(intent, len(x), list(self.known), {"reason_code": r.get("reason_code")}, r.get("reason", ""))
        if not self.online:
            self._queue({"ts": time.time(), "request": json.loads(req.to_json())})
            rec.update(result="QUEUED_OFFLINE", path=["local", "queued"]); self.log.append(rec); return rec
        self.teacher_calls += 1; rec["teacher_called"] = True; rec["path"].append("teacher")
        try:
            resp = parse_response(self.teacher.advise(req), self.known)
        except TeacherUnavailable as e:
            self.teacher_calls -= 1; rec["teacher_called"] = False
            self._queue({"ts": time.time(), "request": json.loads(req.to_json()), "why": f"teacher unavailable: {e}"})
            rec.update(result="QUEUED_OFFLINE", path=["local", "queued"], reason=str(e)); self.log.append(rec); return rec
        except InvalidTeacherResponse as e:
            rec.update(result="NEEDS_HELP", reason=f"teacher response rejected: {e}"); self.log.append(rec); return rec
        rec["teacher_action"] = resp.action
        if resp.action in ("new_capability", "new_capability_spec"):
            heldout = None
            if resp.action == "new_capability_spec":
                if examples is None:
                    rec.update(result="NEEDS_HELP", reason="teacher rule cannot be verified: environment examples required"); self.log.append(rec); return rec
                try:
                    spec = TaskSpec.from_dict(resp.spec)
                    lp = spec_to_learning_package(spec, self.teacher.name)
                    used = set(map(tuple, lp.train_x)) | set(map(tuple, lp.validation_x))
                    heldout = sample_spec(spec, 400 if not all("grid" in d for d in spec.domain) else 80, 7, exclude=used, unique=True)
                except (InvalidSpec, UnsafeExpression) as e:
                    rec.update(result="NEEDS_HELP", reason=f"teacher spec rejected: {e}"); self.log.append(rec); return rec
            else:
                lp = resp.learning_package
            if classify_information(lp.description) == "MEMORY" or lp.info_kind != "CAPABILITY":
                rec.update(result="NEEDS_HELP", reason="teacher proposed neural training for non-capability information"); self.log.append(rec); return rec
            try:
                validate(lp, self.known)
            except InvalidLearningPackage as e:
                rec.update(result="NEEDS_HELP", reason=f"learning package rejected: {e}"); self.log.append(rec); return rec
            ver = self.versions.get(lp.capability_id, 0) + 1
            built = self.service.build(lp, f"0.{ver}.0", self.known, heldout=heldout, verification=examples)
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
            if again["result"] == "ANSWER" and examples is not None and self.persist_router_updates:
                ok, why = self._router_update(intent, resp.capability_id, examples)
                rec["router_update"] = {"accepted": ok, "why": why}
                if ok:
                    rec["path"].append("router_update")
        elif resp.action in ("external_research", "request_tool"):
            # OpenClaw is not integrated before Phase 8: report honestly, do not pretend research happened
            rec.update(result="NEEDS_EXTERNAL", needs=resp.action, detail=resp.research_query or resp.tool)
        else:
            rec.update(result="NEEDS_HELP", reason="teacher cannot help")
        self.log.append(rec)
        return rec


    _STOP = {"a", "an", "the", "of", "to", "and", "or", "is", "are", "two", "this", "that", "please", "for", "in", "on", "these", "those", "with", "from", "which", "what", "given", "me", "it", "my"}

    def _router_update(self, intent: str, cap: str, examples) -> tuple[bool, str]:
        """Cheapest strategy: add the unmatched intent's words to the capability's routing keywords (no neural change).
        Gates: (1) the capability must solve the environment examples; (2) after installing, every previously served intent
        must still route to the same capability, and the new intent must now route KNOWN to `cap`; otherwise roll back."""
        xs, ys = examples
        cases = self.workdir / "verify_cases.jsonl"
        cases.write_text("\n".join(json.dumps({"intent": "x", "input": x, "expected_index": y}) for x, y in zip(xs[:200], ys[:200])) + "\n")
        b = self.cli._run_detect("keyword", "batch", "--cases", str(cases), capability=cap)
        if b["summary"]["accuracy"] < 0.95:
            return False, f"capability fails environment examples ({b['summary']['accuracy']:.3f})"
        words = [w for w in intent.lower().replace(",", " ").split() if w not in self._STOP and w.isalpha()]
        self.versions[cap] = self.versions.get(cap, 0)
        cur = next(c for c in self.cli.list() if c["capability_id"] == cap)["active_version"]
        ma, mi, pa = map(int, cur.split("."))
        built = self.service.repackage(cap, f"{ma}.{mi}.{pa + 1}", extra_keywords=words, strategy="router_update", note=f"alias intent: {intent[:60]}")
        imp = self.cli.import_caps(str(built.cap_path))[0]
        if not imp["activated"]:
            return False, f"import failed at {imp['failed_step']}"
        bad = []
        for i, xx, c in self.routing_suite:
            r2 = self.cli.solve(i, xx)
            if r2.get("capability_id") != c or r2.get("status") != "KNOWN":
                bad.append(i)
        chk = self.cli.solve(intent, xs[0])
        if bad or chk.get("capability_id") != cap or chk.get("status") != "KNOWN":
            self.cli.json("rollback", cap)
            return False, f"routing regression ({len(bad)} previous intents changed; new intent status {chk.get('status')}); rolled back"
        self.router_updates += 1
        return True, f"added keywords {words}"
