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
from integrations.openclaw.action import ActionDenied, NeedsApproval, ActionUnavailable


class Escalator:
    def __init__(self, cli, teacher: TeacherProvider, service, workdir: Path, online=True, max_teacher_calls_per_task=1, broker=None):
        self.cli, self.teacher, self.service, self.broker = cli, teacher, service, broker
        self.workdir = Path(workdir); self.workdir.mkdir(parents=True, exist_ok=True)
        self.online, self.max_calls = online, max_teacher_calls_per_task
        self.known: set[str] = {c["capability_id"] for c in cli.list()}
        self.memory: list[dict] = []
        self.log: list[dict] = []
        self.teacher_calls = 0
        self.openclaw_calls = 0                  # successful external research actions
        self.versions: dict[str, int] = {c["capability_id"]: int(c["active_version"].split(".")[1]) for c in cli.list()}   # survive restarts
        self.routing_suite: list[tuple[str, list[float], str]] = []   # (intent, input, capability) served KNOWN: regression suite for routing changes
        self.router_updates = 0
        self.persist_router_updates = True

    def _queue(self, entry):
        with open(self.workdir / "pending_help.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _ask(self, req, rec):
        """One teacher call. Returns the validated response, or None after filling `rec` with the failure."""
        self.teacher_calls += 1; rec["teacher_called"] = True; rec["teacher_calls"] = rec.get("teacher_calls", 0) + 1
        if "teacher" not in rec["path"]:
            rec["path"].append("teacher")
        try:
            return parse_response(self.teacher.advise(req), self.known)
        except TeacherUnavailable as e:
            self.teacher_calls -= 1; rec["teacher_calls"] -= 1; rec["teacher_called"] = rec["teacher_calls"] > 0
            self._queue({"ts": time.time(), "request": json.loads(req.to_json()), "why": f"teacher unavailable: {e}"})
            rec.update(result="QUEUED_OFFLINE", path=rec["path"] + ["queued"], reason=str(e))
        except InvalidTeacherResponse as e:
            rec.update(result="NEEDS_HELP", reason=f"teacher response rejected: {e}")
        return None

    def solve(self, intent: str, x: list[float], examples: tuple | None = None) -> dict:
        """`examples`: optional (xs, ys) labelled examples supplied by the environment/user, used to verify teacher rules."""
        r = self.cli.solve(intent, x)
        rec = {"intent": intent, "path": ["local"], "teacher_called": False, "task_id": f"esc{len(self.log)}"}
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
        resp = self._ask(req, rec)
        if resp is not None and resp.action == "external_research":
            resp = self._research(resp, req, intent, x, rec)
        if resp is not None:
            self._act(resp, intent, x, examples, rec)
        self.log.append(rec)
        return rec

    # ---- external research through the broker (policy -> provider -> audit) --------------------------------
    def _research(self, resp, req, intent, x, rec):
        if self.broker is None:
            rec.update(result="NEEDS_EXTERNAL", needs="external_research", detail=resp.research_query); return None
        try:
            evidence = self.broker.search(rec["task_id"], resp.research_query)
        except ActionDenied as e:
            rec.update(result="NEEDS_HELP", reason=f"research denied by policy: {e.decision['reason']}", research={"denied": e.decision}); return None
        except NeedsApproval as e:
            rec.update(result="NEEDS_APPROVAL", reason=str(e), research={"approval": e.decision}); return None
        except ActionUnavailable as e:
            self._queue({"ts": time.time(), "request": json.loads(req.to_json()), "why": f"research unavailable: {e.kind}", "query": resp.research_query})
            rec.update(result="QUEUED_EXTERNAL", reason=f"external research unavailable ({e.kind}); request queued, nothing was researched", path=rec["path"] + ["queued"],
                       research={"unavailable": e.kind}); return None
        self.openclaw_calls += 1; rec["openclaw_called"] = True; rec["path"].append("openclaw")
        rec["research"] = {"query": resp.research_query, "evidence": [{"id": e.id, "source": e.source, "sha256": e.sha256, "simulated": e.simulated} for e in evidence],
                           "simulated": any(e.simulated for e in evidence)}
        self._evidence = evidence
        req2 = make_help_request(intent, len(x), list(self.known), req.attempted_route, req.failure, evidence=[e.public() for e in evidence])
        resp2 = self._ask(req2, rec)
        if resp2 is not None and resp2.action == "external_research":
            rec.update(result="NEEDS_HELP", reason="teacher asked for a second research round; limit is one per task"); return None
        return resp2

    # ---- act on a validated teacher response ----------------------------------------------------------------
    def _act(self, resp, intent, x, examples, rec):
        rec["teacher_action"] = resp.action
        evidence = getattr(self, "_evidence", None) if rec.get("openclaw_called") else None
        if resp.action in ("new_capability", "new_capability_spec"):
            heldout = None
            if resp.action == "new_capability_spec":
                if examples is None:
                    rec.update(result="NEEDS_HELP", reason="teacher rule cannot be verified: environment examples required"); return
                try:
                    spec = TaskSpec.from_dict(resp.spec)
                    lp = spec_to_learning_package(spec, self.teacher.name)
                    used = set(map(tuple, lp.train_x)) | set(map(tuple, lp.validation_x))
                    heldout = sample_spec(spec, 400 if not all("grid" in d for d in spec.domain) else 80, 7, exclude=used, unique=True)
                except (InvalidSpec, UnsafeExpression) as e:
                    rec.update(result="NEEDS_HELP", reason=f"teacher spec rejected: {e}"); return
            else:
                lp = resp.learning_package
            if evidence:   # preserve provenance; internet content is never ground truth
                lp.provenance["evidence"] = [{"id": e.id, "provider": e.provider, "source": e.source, "retrieved_at": e.retrieved_at, "sha256": e.sha256, "simulated": e.simulated} for e in evidence]
                lp.provenance["unverified_internet_content"] = True
            if classify_information(lp.description) == "MEMORY" or lp.info_kind != "CAPABILITY":
                rec.update(result="NEEDS_HELP", reason="teacher proposed neural training for non-capability information"); return
            try:
                validate(lp, self.known)
            except InvalidLearningPackage as e:
                rec.update(result="NEEDS_HELP", reason=f"learning package rejected: {e}"); return
            ver = self.versions.get(lp.capability_id, 0) + 1
            built = self.service.build(lp, f"0.{ver}.0", self.known, heldout=heldout, verification=examples)
            if not built.promoted:
                rec.update(result="NEEDS_HELP", reason=built.reason); return
            imp = self.cli.import_caps(str(built.cap_path))[0]
            if not imp["activated"]:
                rec.update(result="NEEDS_HELP", reason=f"import failed at {imp['failed_step']}"); return
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
        elif resp.action == "request_tool":
            if self.broker is None:
                rec.update(result="NEEDS_EXTERNAL", needs="request_tool", detail=resp.tool); return
            dec = self.broker.check(rec["task_id"], f"tool.{resp.tool}", {})
            self.broker._audit(task=rec["task_id"], action=f"tool.{resp.tool}", decision=dec["effect"], rule=dec["rule_id"], reason=dec["reason"], result="not_executed")
            if dec["effect"] == "deny":
                rec.update(result="NEEDS_HELP", reason=f"tool '{resp.tool}' denied by policy: {dec['reason']}")
            else:
                rec.update(result="NEEDS_EXTERNAL", needs="request_tool", detail=resp.tool, reason="tool execution is not implemented yet (Phase 13)")
        elif resp.action == "external_research":
            rec.update(result="NEEDS_EXTERNAL", needs="external_research", detail=resp.research_query)
        else:
            rec.update(result="NEEDS_HELP", reason="teacher cannot help")

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
