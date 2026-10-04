"""Desktop application: a configurable local service around the Rust core.
  python3 -m apps.desktop.app init  --dir STATE            create state dir, signing key, trust root, API token, default config
  python3 -m apps.desktop.app serve --config STATE/desktop.json   local JSON API on 127.0.0.1 (bearer token)
  python3 -m apps.desktop.app chat  --config STATE/desktop.json   small terminal REPL
Single-user "local trust" mode: this process holds the signing key (a production deployment would keep the build/signing service separate)."""
from __future__ import annotations
import argparse, hmac, json, os, secrets, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.cli import Cli, AICLI
from packages.capbuild import keygen, write_trust
from integrations.escalation import Escalator
from integrations.teacher.simulator import TeacherSimulator
from integrations.teacher.http_providers import OpenAICompatProvider, AnthropicProvider
from integrations.openclaw.broker import ActionBroker
from integrations.openclaw.openclaw_cli import OpenClawCli, OpenClawActionProvider, OpenClawModelTeacher
from training.service import BuildService
from training.learning_package.examples import package_from_examples
from training.learning_package.schema import InvalidLearningPackage

MAX_BODY = 5 * 1024 * 1024


class NullOracle:
    def heldout(self, *a, **k):
        raise RuntimeError("no oracle available: provide explicit held-out examples")


def default_config(d: Path) -> dict:
    return {"root": str(d / "runtime"), "trust": str(d / "trust.json"), "keys_dir": str(d / "keys"), "signer_id": "local-build-1", "device": "PC_FULL",
            "teacher": {"kind": "none"}, "policy": str(ROOT / "integrations/openclaw/policy.default.json"), "openclaw": None,
            "api": {"host": "127.0.0.1", "port": 8787, "token_file": str(d / "api.token")}}


def init(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    cfg = default_config(d)
    if not (d / "keys" / "local-build-1.private").exists():
        keygen("local-build-1", d / "keys")
    write_trust(d / "trust.json", {"local-build-1": (d / "keys/local-build-1.public").read_text()})
    tok = d / "api.token"
    if not tok.exists():
        tok.write_text(secrets.token_hex(32)); tok.chmod(0o600)
    (d / "desktop.json").write_text(json.dumps(cfg, indent=2))
    return d / "desktop.json"


class DesktopApp:
    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.cfg = json.loads(self.config_path.read_text())
        self.lock = threading.Lock()
        root = Path(self.cfg["root"]); root.mkdir(parents=True, exist_ok=True)
        self.cli = Cli(root, Path(self.cfg["trust"]), self.cfg.get("device", "PC_FULL"))
        self.teacher = self._teacher(self.cfg.get("teacher", {"kind": "none"}))
        self.svc = BuildService(Path(self.cfg["keys_dir"]), self.cfg["signer_id"], root / "build", NullOracle())
        provider = None
        if self.cfg.get("openclaw"):
            oc = self.cfg["openclaw"]; provider = OpenClawActionProvider(OpenClawCli(command=oc.get("command"), search_provider=oc.get("search_provider"), state_dir=root / "openclaw-state"))
        self.broker = ActionBroker(AICLI, Path(self.cfg["policy"]), provider, root / "audit.jsonl", root / "approvals.json") if self.cfg.get("policy") else None
        self.esc = Escalator(self.cli, self.teacher, self.svc, root / "escalation", broker=self.broker) if self.teacher else None

    def _teacher(self, t):
        k = t.get("kind", "none")
        if k == "none":
            return None
        if k == "simulator-spec":
            return TeacherSimulator(mode="spec")
        if k == "openai-compat":
            return OpenAICompatProvider(t["base_url"], t["model"], api_key_env=t.get("api_key_env", "OPENAI_API_KEY"))
        if k == "anthropic":
            return AnthropicProvider(t["model"], api_key_env=t.get("api_key_env", "ANTHROPIC_API_KEY"))
        if k == "openclaw-model":
            return OpenClawModelTeacher(OpenClawCli(command=t.get("command"), state_dir=Path(self.cfg["root"]) / "openclaw-state"), t["model"])
        raise ValueError(f"unknown teacher kind {k}")

    # ---- operations (all serialised: the registry is a single file) ----
    def status(self):
        with self.lock:
            caps = self.cli.list()
            return {"capabilities": [{k: c[k] for k in ("capability_id", "active_version", "params", "model_bytes", "calls", "input_dim")} for c in caps],
                    "teacher": getattr(self.teacher, "name", None), "device": self.cfg.get("device"), "escalation": None if not self.esc else
                    {"teacher_calls": self.esc.teacher_calls, "openclaw_calls": self.esc.openclaw_calls, "tasks": len(self.esc.log)}}

    def solve(self, intent: str, x: list[float], examples=None):
        with self.lock:
            if self.esc:
                return self.esc.solve(intent, x, examples=tuple(examples) if examples else None)
            return self.cli.solve(intent, x)

    def teach(self, spec: dict) -> dict:
        """Learn a capability from labelled examples supplied by the user (no teacher involved)."""
        cap, xs, ys = spec["capability_id"], spec["x"], spec["y"]
        labels = spec.get("labels") or [str(i) for i in range(max(ys) + 1)]
        kws = spec.get("keywords") or sorted({w for w in (spec.get("description", "") + " " + cap.replace("_", " ")).lower().split() if len(w) > 2})
        with self.lock:
            minor = int(next((c["active_version"] for c in self.cli.list() if c["capability_id"] == cap), "0.0.0").split(".")[1])
            try:
                lp, held, info = package_from_examples(cap, spec.get("description", cap), kws, labels, xs, ys, seed=int(spec.get("seed", 0)))
            except ValueError as e:                      # e.g. too few distinct examples per class for a stratified split
                return {"learned": False, "reason": f"invalid examples: {str(e)[:160]}"}
            try:
                b = self.svc.build(lp, f"0.{minor + 1}.0", {c["capability_id"] for c in self.cli.list()}, heldout=held, standardize=True, min_acc=float(spec.get("min_acc", 0.9)),
                                   attempts=((32, 32), (64, 64), (128, 64)))
            except InvalidLearningPackage as e:
                return {"learned": False, "reason": f"invalid examples: {e}"}
            if not b.promoted:
                return {"learned": False, "reason": b.reason, "data": info}
            imp = self.cli.import_caps(str(b.cap_path))[0]
            return {"learned": imp["activated"], "version": f"0.{minor + 1}.0", "data": info, "heldout_accuracy": b.report["attempts"][-1]["heldout_accuracy"],
                    "params": b.report["attempts"][-1]["params"], "failed_step": imp.get("failed_step"), "teacher_used": False}


def make_handler(app: DesktopApp, token: str, port: int):
    class H(BaseHTTPRequestHandler):
        server_version = "aiagent-desktop"
        def log_message(self, *a): pass

        def _send(self, code: int, obj):
            data = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(data)

        def _guard(self, need_auth=True) -> bool:
            host = (self.headers.get("Host") or "").lower()
            if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                self._send(403, {"error": "bad host header"}); return False        # DNS-rebinding defence
            if self.headers.get("Origin"):
                self._send(403, {"error": "browser cross-origin requests are not accepted"}); return False
            if need_auth:
                got = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
                if not hmac.compare_digest(got.encode(), token.encode()):
                    self._send(401, {"error": "unauthorized"}); return False
            return True

        def do_GET(self):
            if self.path == "/health":
                if self._guard(need_auth=False): self._send(200, {"ok": True})
                return
            if not self._guard(): return
            if self.path == "/status": self._send(200, app.status())
            elif self.path == "/capabilities": self._send(200, app.status()["capabilities"])
            else: self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self._guard(): return
            if self.path not in ("/solve", "/teach"):
                self._send(404, {"error": "not found"}); return
            if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
                self._send(415, {"error": "content-type must be application/json"}); return
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._send(400, {"error": "bad content-length"}); return
            if n <= 0 or n > MAX_BODY:
                self._send(413 if n > MAX_BODY else 400, {"error": "body size not acceptable"}); return
            try:
                body = json.loads(self.rfile.read(n))
                assert isinstance(body, dict)
            except Exception:
                self._send(400, {"error": "malformed JSON"}); return
            try:
                if self.path == "/solve":
                    x = body["input"]
                    if not isinstance(body.get("intent"), str) or not isinstance(x, list) or not all(isinstance(v, (int, float)) for v in x) or len(x) > 4096:
                        raise ValueError("intent (string) and input (list of numbers) required")
                    self._send(200, app.solve(body["intent"], x, (body["examples"]["x"], body["examples"]["y"]) if body.get("examples") else None))
                else:
                    for k in ("capability_id", "x", "y"):
                        if k not in body: raise ValueError(f"missing field {k}")
                    if not isinstance(body["capability_id"], str) or not body["capability_id"].replace("_", "").isalnum() or len(body["x"]) != len(body["y"]) or len(body["x"]) > 50000:
                        raise ValueError("invalid capability_id or example arrays")
                    self._send(200, app.teach(body))
            except (KeyError, ValueError, TypeError) as e:
                self._send(400, {"error": str(e)[:200]})
            except Exception as e:                       # never leak stack traces
                self._send(500, {"error": f"internal error ({type(e).__name__})"})
    return H


def serve(app: DesktopApp, port: int | None = None, token: str | None = None) -> HTTPServer:
    api = app.cfg["api"]
    port = api["port"] if port is None else port
    token = token or Path(api["token_file"]).read_text().strip()
    srv = HTTPServer((api["host"], port), None)
    srv.RequestHandlerClass = make_handler(app, token, srv.server_address[1])
    return srv


def chat(app: DesktopApp):
    print("commands: /caps | /solve <intent> :: x1,x2,... | /quit")
    for line in sys.stdin:
        line = line.strip()
        if line == "/quit": break
        if line == "/caps": print(json.dumps(app.status()["capabilities"], indent=1))
        elif line.startswith("/solve ") and "::" in line:
            intent, xs = line[7:].split("::", 1)
            print(json.dumps(app.solve(intent.strip(), [float(v) for v in xs.split(",")]), indent=1))
        else: print("unknown command")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("init"); a.add_argument("--dir", required=True)
    for n in ("serve", "chat"):
        b = sub.add_parser(n); b.add_argument("--config", required=True)
    args = ap.parse_args()
    if args.cmd == "init":
        print(init(Path(args.dir)))
    elif args.cmd == "serve":
        app = DesktopApp(Path(args.config)); srv = serve(app); print(f"listening on http://{srv.server_address[0]}:{srv.server_address[1]} (token in {app.cfg['api']['token_file']})"); srv.serve_forever()
    else:
        chat(DesktopApp(Path(args.config)))


if __name__ == "__main__":
    main()
