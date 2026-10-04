"""ActionBroker: the single door to external actions. policy check (Rust, deterministic) -> provider -> audit log.
Privacy: queries are redacted before they are even evaluated or sent. The broker has no method that changes policy or approvals."""
from __future__ import annotations
import hashlib, json, subprocess, time
from pathlib import Path
from integrations.teacher.provider import redact
from integrations.openclaw.action import ActionProvider, ActionDenied, NeedsApproval, ActionUnavailable, Evidence

MAX_EVIDENCE = 5


class ActionBroker:
    def __init__(self, aicli: Path, policy_path: Path, provider: ActionProvider | None, audit_path: Path, approvals_path: Path | None = None):
        self.aicli, self.policy_path, self.provider = Path(aicli), Path(policy_path), provider
        self.audit_path, self.approvals_path = Path(audit_path), (Path(approvals_path) if approvals_path else None)
        self.counts: dict[str, dict[str, int]] = {}      # task_id -> action -> count
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    # -- policy -------------------------------------------------------------------------------------------------
    def check(self, task_id: str, action: str, params: dict) -> dict:
        req = {"action": action, "params": params, "task_id": task_id, "counts": self.counts.get(task_id, {})}
        cmd = [str(self.aicli), "--root", "/nonexistent", "--trust", "/nonexistent", "policy-check", "--policy", str(self.policy_path), "--request", json.dumps(req)]
        if self.approvals_path:
            cmd += ["--approvals", str(self.approvals_path)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            return {"effect": "deny", "rule_id": "policy-engine-error", "reason": p.stderr.strip()[:200]}   # fail closed
        return json.loads(p.stdout)

    def audit(self, **kw):
        """Append an audit line (public so other action surfaces, e.g. the tool host, share one log)."""
        self._audit(**kw)

    def _audit(self, **kw):
        with open(self.audit_path, "a") as f:
            f.write(json.dumps({"ts": time.time(), **kw}) + "\n")

    def _run(self, task_id: str, action: str, params: dict, call) -> list[Evidence]:
        dec = self.check(task_id, action, params)
        base = {"task": task_id, "action": action, "params_sha256": hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest(),
                "params_preview": {k: str(v)[:80] for k, v in params.items()}, "decision": dec["effect"], "rule": dec["rule_id"], "reason": dec["reason"]}
        if dec["effect"] == "deny":
            self._audit(**base, result="denied"); raise ActionDenied(dec)
        if dec["effect"] == "require_approval":
            self._audit(**base, result="needs_approval"); raise NeedsApproval(dec)
        if self.provider is None:
            self._audit(**base, result="unavailable:no_provider"); raise ActionUnavailable("no_provider", "no action provider configured")
        self.counts.setdefault(task_id, {})[action] = self.counts.get(task_id, {}).get(action, 0) + 1
        try:
            ev = call()[:MAX_EVIDENCE]
        except ActionUnavailable as e:
            self._audit(**base, result=f"unavailable:{e.kind}", detail=e.message[:160]); raise
        self._audit(**base, result="ok", provider=self.provider.name, evidence=[{"id": e.id, "sha256": e.sha256, "source": e.source, "simulated": e.simulated} for e in ev])
        return ev

    # -- actions ------------------------------------------------------------------------------------------------
    def search(self, task_id: str, query: str, limit: int = 3) -> list[Evidence]:
        q = redact(query, 200)
        return self._run(task_id, "web.search", {"query": q}, lambda: self.provider.search(q, limit))

    def fetch(self, task_id: str, url: str) -> list[Evidence]:
        return self._run(task_id, "web.fetch", {"url": url}, lambda: self.provider.fetch(url))
