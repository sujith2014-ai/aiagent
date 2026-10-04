"""Local mock of an OpenAI-compatible and Anthropic-style endpoint, backed by any TeacherProvider (e.g. the simulator).
Dev/test tool only: lets the real HTTP provider code be exercised without network egress or API keys."""
from __future__ import annotations
import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from integrations.teacher.provider import HelpRequest


def start(backend, fail: bool = False, garbage: str | None = None):
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n))
            seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
            if fail:
                self.send_response(503); self.end_headers(); return
            user = body["messages"][-1]["content"]
            req = HelpRequest(**json.loads(user))
            text = garbage if garbage is not None else backend.advise(req)
            if self.path.endswith("/chat/completions"):
                out = {"choices": [{"message": {"role": "assistant", "content": text}}]}
            else:
                out = {"content": [{"type": "text", "text": text}]}
            data = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}", seen
