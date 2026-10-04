"""Trusted side of tool growth (holds the install-signing key; lives in apps/, never under integrations/ where generated code and teachers are). Installed tools: signed install records, verified on every call; the host runs tools only through the sandbox and the policy engine."""
from __future__ import annotations
import base64, hashlib, json, time
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
from integrations.toolgrowth.package import Candidate, sha, request_hash_of
from integrations.toolgrowth.pipeline import Pipeline, Report
from integrations.toolgrowth.sandbox import SandboxConfig, run_batch


def _ver(v: str): return tuple(int(x) for x in v.split("."))


class ToolRegistry:
    def __init__(self, root: Path, pipeline: Pipeline):
        self.root, self.pipeline = Path(root), pipeline
        (self.root / "tools").mkdir(parents=True, exist_ok=True)
        self.key_file, self.pub_file = self.root / "registry.key", self.root / "registry.pub"
        if not self.key_file.exists():
            k = Ed25519PrivateKey.generate()
            self.key_file.write_bytes(k.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())); self.key_file.chmod(0o600)
            self.pub_file.write_bytes(k.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw))

    # -- signing ------------------------------------------------------------------------------------------------
    def _sign(self, record: dict) -> dict:
        body = json.dumps(record, sort_keys=True).encode()
        sig = Ed25519PrivateKey.from_private_bytes(self.key_file.read_bytes()).sign(body)
        return {"record": record, "signature_b64": base64.b64encode(sig).decode()}

    def _verify(self, doc: dict) -> dict:
        body = json.dumps(doc["record"], sort_keys=True).encode()
        try: Ed25519PublicKey.from_public_bytes(self.pub_file.read_bytes()).verify(base64.b64decode(doc["signature_b64"]), body)
        except (InvalidSignature, KeyError, ValueError): raise ValueError("install record signature is invalid")
        return doc["record"]

    # -- queries ------------------------------------------------------------------------------------------------
    def _dir(self, tool_id: str, version: str) -> Path: return self.root / "tools" / tool_id / version
    def _active_file(self, tool_id: str) -> Path: return self.root / "tools" / tool_id / "ACTIVE"

    def versions(self, tool_id: str) -> list[str]:
        d = self.root / "tools" / tool_id
        return sorted([p.name for p in d.iterdir() if p.is_dir()], key=_ver) if d.exists() else []

    def active_version(self, tool_id: str) -> str | None:
        f = self._active_file(tool_id)
        return f.read_text().strip() if f.exists() else None

    def installed(self) -> list[dict]:
        out = []
        for d in sorted((self.root / "tools").iterdir()):
            v = self.active_version(d.name) if d.is_dir() else None
            if v:
                try: out.append(self.record(d.name, v))
                except Exception: pass                   # an unverifiable tool is not listed
        return out

    def record(self, tool_id: str, version: str) -> dict:
        return self._verify(json.loads((self._dir(tool_id, version) / "installed.json").read_text()))

    # -- install (the only write path) ------------------------------------------------------------------------------
    def install(self, cand: Candidate, spec_cases: list, task_id: str = "tool-growth") -> dict:
        """Re-runs the WHOLE pipeline on exactly this candidate (no trusted earlier report) and installs only if the operator's approval matches its hash."""
        rep = self.pipeline.evaluate(cand, spec_cases, task_id)
        if rep.status != "APPROVED": return {"installed": False, "reason": rep.failed_stage or "awaiting operator approval", "report": rep.public()}
        cur = self.active_version(cand.tool_id)
        if cur and _ver(cand.version) <= _ver(cur): return {"installed": False, "reason": f"version {cand.version} is not newer than installed {cur}", "report": rep.public()}
        d = self._dir(cand.tool_id, cand.version)
        if d.exists(): return {"installed": False, "reason": "version already exists", "report": rep.public()}
        d.mkdir(parents=True)
        (d / "tool.py").write_text(cand.source); (d / "tests.jsonl").write_text(cand.tests_text()); (d / "tool.json").write_text(json.dumps(cand.manifest(), indent=1, sort_keys=True))
        rec = {**cand.manifest(), "approved_request_hash": rep.request_hash, "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "previous_version": cur}
        (d / "installed.json").write_text(json.dumps(self._sign(rec), indent=1, sort_keys=True))
        self._active_file(cand.tool_id).write_text(cand.version)
        return {"installed": True, "tool_id": cand.tool_id, "version": cand.version, "request_hash": rep.request_hash, "report": rep.public()}

    def rollback(self, tool_id: str) -> str | None:
        vs, cur = self.versions(tool_id), self.active_version(tool_id)
        prev = [v for v in vs if cur and _ver(v) < _ver(cur)]
        if not prev: return None
        self._active_file(tool_id).write_text(prev[-1]); return prev[-1]


class ToolHost:
    """Calls installed tools. Every call: policy check, signature + hash verification, sandboxed run with NO permissions, output size cap, audit line."""
    MAX_OUT = 64_000

    def __init__(self, registry: ToolRegistry, broker, cfg: SandboxConfig | None = None):
        self.reg, self.broker, self.cfg = registry, broker, cfg or SandboxConfig()

    def call(self, tool_id: str, inp: dict, task_id: str = "tool-call") -> dict:
        dec = self.broker.check(task_id, "tool.run", {"tool_id": tool_id})
        self.broker.audit(task=task_id, action="tool.run", params_preview={"tool_id": tool_id}, decision=dec["effect"], rule=dec["rule_id"], reason=dec["reason"])
        if dec["effect"] != "allow": return {"ok": False, "refused": "POLICY", "reason": dec["reason"]}
        self.broker.counts.setdefault(task_id, {})["tool.run"] = self.broker.counts.get(task_id, {}).get("tool.run", 0) + 1
        v = self.reg.active_version(tool_id)
        if not v: return {"ok": False, "refused": "NO_SUCH_TOOL", "reason": f"no installed tool {tool_id}"}
        try:
            rec = self.reg.record(tool_id, v)
            src = (self.reg._dir(tool_id, v) / "tool.py").read_text()
            if sha(src) != rec["code_sha256"]: raise ValueError("installed code does not match its signed record")
        except Exception as e:
            self.broker.audit(task=task_id, action="tool.run", result="refused:tampered", tool=tool_id, reason=str(e)[:160])
            return {"ok": False, "refused": "TOOL_UNAVAILABLE", "reason": str(e)}
        r = run_batch(src, [inp], self.cfg)
        if not r.ok: res = {"ok": False, "refused": "SANDBOX", "kind": r.kind, "reason": r.detail}
        elif not r.results[0]["ok"]: res = {"ok": False, "refused": "TOOL_ERROR", "reason": r.results[0]["error"]}
        else:
            out = r.results[0]["output"]
            res = {"ok": True, "output": out, "tool_id": tool_id, "version": v} if isinstance(out, dict) and len(json.dumps(out)) <= self.MAX_OUT else {"ok": False, "refused": "BAD_OUTPUT", "reason": "output is not a small JSON object"}
        self.broker.audit(task=task_id, action="tool.run", result="ok" if res["ok"] else f"failed:{res['refused']}", tool=tool_id, version=v, seconds=round(r.seconds, 4))
        return res


class ToolIndex:
    """Gap detection: which installed tool (if any) covers an intent. Needs >= 2 keyword hits and a unique best tool, otherwise it reports a gap."""
    def __init__(self, registry: ToolRegistry): self.reg = registry

    def match(self, intent: str) -> str | None:
        words = {w for w in intent.lower().replace("?", " ").replace(",", " ").split()}
        scored = sorted(((len(words & set(r["keywords"])), r["tool_id"]) for r in self.reg.installed()), reverse=True)
        if not scored or scored[0][0] < 2 or (len(scored) > 1 and scored[1][0] == scored[0][0]): return None
        return scored[0][1]
