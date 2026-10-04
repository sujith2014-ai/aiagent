"""Shared harness for Phase 13 tests/experiments: a broker + pipeline + registry + host in a temp dir, with an `operator` that approves through the real operator-only `aicli approve`."""
import json, subprocess
from pathlib import Path
from integrations.openclaw.broker import ActionBroker
from integrations.toolgrowth.pipeline import Pipeline
from apps.toolhost.registry import ToolRegistry, ToolHost, ToolIndex
from integrations.toolgrowth.sandbox import SandboxConfig
from integrations.toolgrowth.package import Candidate
from scripts.cli import AICLI, ROOT

POLICY = ROOT / "integrations/toolgrowth/policy.tools.json"


class ToolLab:
    def __init__(self, tmp: Path, reserved=None, cfg: SandboxConfig | None = None, static=True):
        self.tmp = Path(tmp); self.tmp.mkdir(parents=True, exist_ok=True)
        self.approvals = self.tmp / "approvals.json"; self.audit = self.tmp / "audit.jsonl"
        self.broker = ActionBroker(AICLI, POLICY, None, self.audit, self.approvals)
        self.pipeline = Pipeline(self.broker, reserved or [], cfg, static=static)
        self.registry = ToolRegistry(self.tmp / "registry", self.pipeline)
        self.host = ToolHost(self.registry, self.broker, cfg)
        self.index = ToolIndex(self.registry)

    def operator_approve(self, cand: Candidate, ttl=300, params: dict | None = None):
        """The ONLY way an approval comes into existence: the operator-only CLI command (nothing in the Python tool-growth code can create one)."""
        req = {"action": "tool.install", "params": params if params is not None else cand.approval_params()}
        p = subprocess.run([str(AICLI), "--root", "/x", "--trust", "/x", "approve", "--approvals", str(self.approvals), "--request", json.dumps(req), "--approver", "operator", "--ttl", str(ttl)], capture_output=True, text=True)
        assert p.returncode == 0, p.stderr
        return json.loads(p.stdout)

    def audit_lines(self):
        return [json.loads(l) for l in self.audit.read_text().splitlines()] if self.audit.exists() else []
