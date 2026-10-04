"""Shared harness for the Phase 11 tests and experiment: a capability server, a device runtime and a hybrid client wired together on loopback."""
import json, threading, urllib.request, urllib.error
from pathlib import Path
from sklearn.datasets import load_iris
from apps.server.capserver import init, CapServer, serve
from apps.hybrid.client import HybridClient
from scripts.cli import Cli

IRIS_INTENT = "classify iris flower species from measurements"


def iris_spec(**over):
    d = load_iris()
    return {"capability_id": "iris_species", "description": IRIS_INTENT, "labels": [str(t) for t in d.target_names], "x": d.data.tolist(), "y": d.target.tolist(), "min_acc": 0.9, **over}


class Lab:
    def __init__(self, tmp, device="PC_CONSTRAINED", max_model_bytes=8 << 20, **client_kw):
        self.tmp = tmp
        init(tmp / "srv"); self.app = CapServer(tmp / "srv"); self.start()
        self.token = (tmp / "srv/server.token").read_text().strip()
        self.dev_trust = tmp / "dev_trust.json"; self.dev_trust.write_text((tmp / "srv/trust.json").read_text())      # public keys only
        self.local = Cli(tmp / "dev/rt", self.dev_trust, device)
        self.client = HybridClient(self.local, f"http://127.0.0.1:{self.port}", self.token, tmp / "dev/work", self.dev_trust, max_model_bytes, **client_kw)

    def start(self, port=0):
        self.srv = serve(self.app, port=port); self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True); self.thread.start()

    def stop(self):
        self.srv.shutdown(); self.srv.server_close()

    def raw(self, method, path, body=None, headers=None, host=None, auth=True):
        h = {"Host": host or f"127.0.0.1:{self.port}", **(headers or {})}
        if auth: h["Authorization"] = f"Bearer {self.token}"
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as r: return r.status, r.read()
        except urllib.error.HTTPError as e: return e.code, e.read()


