"""Adapter over the OpenClaw CLI's narrow, non-agentic `infer` surface (web search/fetch, one-shot model run).
No autonomous agent turn is ever started and no OpenClaw tool/shell/browser capability is used.
Validated against a real install (OpenClaw 2026.6.35, Node 22.22) for: the CLI contract (flags, exit codes: failures exit 1 with the message
on stderr and empty stdout; success exits 0 with JSON on stdout) and the provider listing output. NOT validated live: search/fetch success
output (the egress proxy denied the search provider; no model credentials). The result parsing below follows the envelope in OpenClaw's own
source (`{"ok":true,"capability":...,"provider":...,"outputs":[{"result":...}]}`) and is defensive about the provider-specific `result`."""
from __future__ import annotations
import json, os, subprocess, tempfile
from pathlib import Path
from integrations.openclaw.action import ActionProvider, ActionUnavailable, Evidence, make_evidence
from integrations.teacher.provider import TeacherProvider, HelpRequest
from integrations.teacher.http_providers import SYSTEM_PROMPT, TeacherUnavailable

MAX_STDOUT = 400_000


def classify_failure(stderr: str) -> str:
    s = stderr.lower()
    if "no provider is available" in s or "is disabled" in s or "not available" in s:
        return "no_provider"
    if "proxy response" in s or "fetch failed" in s or "econn" in s or "enotfound" in s or "network" in s:
        return "network"
    if "api key" in s or "unauthorized" in s or "auth" in s:
        return "auth"
    return "other"


def _hits(node, out):
    """Collect dicts that look like search hits / page content anywhere inside a provider result."""
    if isinstance(node, dict):
        if any(k in node for k in ("url", "link")) and any(k in node for k in ("title", "snippet", "description", "content", "text")):
            out.append(node)
        for v in node.values():
            _hits(v, out)
    elif isinstance(node, list):
        for v in node:
            _hits(v, out)


class OpenClawCli:
    def __init__(self, command: list[str] | None = None, profile: str = "aiagent", state_dir: Path | None = None, timeout: float = 60.0, search_provider: str | None = None):
        self.command = command or ["openclaw"]
        self.profile, self.timeout, self.search_provider = profile, timeout, search_provider
        self.state_dir = Path(state_dir or tempfile.mkdtemp(prefix="openclaw-state-"))

    def _env(self) -> dict:
        # minimal environment: nothing from the caller's environment (no API keys, no tokens) is inherited, except proxy settings
        keep = {k: v for k, v in os.environ.items() if k in ("PATH", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE")}
        return {**keep, "HOME": str(self.state_dir), "OPENCLAW_HOME": str(self.state_dir)}

    def run(self, *args: str) -> dict:
        cmd = [*self.command, "--profile", self.profile, *args]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, env=self._env())
        except subprocess.TimeoutExpired:
            raise ActionUnavailable("timeout", f"no answer within {self.timeout}s")
        except FileNotFoundError:
            raise ActionUnavailable("not_installed", f"{self.command[0]} not found")
        if p.returncode != 0:
            raise ActionUnavailable(classify_failure(p.stderr), (p.stderr.strip() or f"exit code {p.returncode}")[:300])
        if len(p.stdout) > MAX_STDOUT:
            raise ActionUnavailable("too_large", f"output exceeds {MAX_STDOUT} bytes")
        try:
            env = json.loads(p.stdout)
        except json.JSONDecodeError:
            raise ActionUnavailable("bad_output", "stdout is not JSON")
        if not isinstance(env, dict) or env.get("ok") is False:
            raise ActionUnavailable("other", str(env)[:200])
        return env

    def providers(self) -> dict:
        return self.run("infer", "web", "providers", "--json")


class OpenClawActionProvider(ActionProvider):
    def __init__(self, cli: OpenClawCli):
        self.cli = cli
        self.name = "openclaw" + (f":{cli.search_provider}" if cli.search_provider else "")

    def search(self, query: str, limit: int = 3):
        args = ["infer", "web", "search", "--query", query, "--limit", str(limit), "--json"]
        if self.cli.search_provider:
            args += ["--provider", self.cli.search_provider]
        env = self.cli.run(*args)
        hits: list[dict] = []
        _hits(env.get("outputs", []), hits)
        if not hits:
            raise ActionUnavailable("no_results", "provider returned no recognisable results")
        return [make_evidence(i, f"openclaw:{env.get('provider', '?')}", f"search:{query}|{h.get('url') or h.get('link')}",
                              " ".join(str(h.get(k, "")) for k in ("title", "snippet", "description", "content", "text") if h.get(k))) for i, h in enumerate(hits[:limit])]

    def fetch(self, url: str):
        env = self.cli.run("infer", "web", "fetch", "--url", url, "--json")
        res = (env.get("outputs") or [{}])[0].get("result", {})
        text = ""
        if isinstance(res, str):
            text = res
        elif isinstance(res, dict):
            text = next((str(res[k]) for k in ("text", "content", "markdown", "body") if res.get(k)), json.dumps(res)[:5000])
        if not text:
            raise ActionUnavailable("no_results", "empty page")
        return [make_evidence(0, f"openclaw:{env.get('provider', '?')}", url, text)]


class OpenClawModelTeacher(TeacherProvider):
    """A teacher reached through OpenClaw's lean one-shot `infer model run` (no agent turn, no tools). Live validation PENDING (needs model auth)."""
    def __init__(self, cli: OpenClawCli, model: str):
        self.cli, self.model = cli, model
        self.name = f"openclaw-model:{model}"

    def advise(self, req: HelpRequest) -> str:
        prompt = SYSTEM_PROMPT + "\n\nREQUEST:\n" + req.to_json()
        try:
            env = self.cli.run("infer", "model", "run", "--local", "--model", self.model, "--prompt", prompt, "--json")
        except ActionUnavailable as e:
            raise TeacherUnavailable(f"{e.kind}: {e.message}")
        outs = env.get("outputs") or [{}]
        res = outs[0].get("result", outs[0])
        if isinstance(res, str):
            return res
        for k in ("text", "output", "content", "reply"):
            if isinstance(res, dict) and isinstance(res.get(k), str):
                return res[k]
        raise TeacherUnavailable("could not find text in model output")
