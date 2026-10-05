"""Thin driver over `openclaw browser ...` (the supported OpenClaw browser surface). It runs the CLI as a subprocess against a local OpenClaw gateway.
It never types credentials, never reads cookies/storage, and exposes only the few actions the browser teacher needs."""
from __future__ import annotations
import json, os, re, subprocess
from dataclasses import dataclass


class BrowserError(Exception):
    pass


@dataclass
class Ref:
    role: str
    name: str
    ref: str


_REF = re.compile(r'^\s*-\s+([A-Za-z]+)(?:\s+"((?:[^"\\]|\\.)*)")?[^\n]*?\[ref=(e\d+)\]', re.M)


def parse_snapshot(text: str) -> list[Ref]:
    return [Ref(m.group(1), (m.group(2) or ""), m.group(3)) for m in _REF.finditer(text)]


def _json_from(out: str):
    out = out.strip()
    try: return json.loads(out)
    except ValueError: pass
    for i, line in enumerate(out.splitlines()):
        if line[:1] in "[{\"" or line[:1].isdigit() or line.startswith(("true", "false", "null")):
            try: return json.loads("\n".join(out.splitlines()[i:]))
            except ValueError: continue
    raise BrowserError(f"no JSON in browser output: {out[:120]!r}")


class OpenClawBrowser:
    def __init__(self, command: list[str], profile: str = "openclaw", home: str | None = None, gateway_token: str | None = None, timeout_s: float = 60.0):
        self.command, self.profile, self.timeout_s = list(command), profile, timeout_s
        self.env = {**os.environ, **({"HOME": home} if home else {}), **({"OPENCLAW_GATEWAY_TOKEN": gateway_token} if gateway_token else {})}

    def _run(self, *args: str) -> str:
        cmd = self.command + ["browser", "--browser-profile", self.profile, "--timeout", str(int(self.timeout_s * 1000)), *args]
        try: p = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s + 15, env=self.env)
        except subprocess.TimeoutExpired: raise BrowserError(f"timeout running browser {args[0]}")
        if p.returncode != 0: raise BrowserError((p.stderr or p.stdout).strip()[-300:])
        return p.stdout

    def start(self): self._run("start")

    def open(self, url: str) -> str:
        out = self._run("open", url); m = re.search(r"^tab:\s*(\S+)", out, re.M)
        return m.group(1) if m else ""

    def snapshot(self) -> str: return self._run("snapshot")
    def type(self, ref: str, text: str): self._run("type", ref, text)
    def click(self, ref: str): self._run("click", ref)
    def press(self, key: str): self._run("press", key)
    def wait_ms(self, ms: int): self._run("wait", "--time", str(ms))
    def close(self, tab: str):
        try: self._run("close", tab)
        except BrowserError: pass

    def evaluate(self, fn: str):
        return _json_from(self._run("evaluate", "--fn", fn))
