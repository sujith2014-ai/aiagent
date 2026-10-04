"""Capability server (PC/server side): holds the signing key, trains from uploaded examples, signs packages, serves a signed catalog,
and can execute tasks for devices that cannot run a capability locally. Devices hold only public trust roots.
  python3 -m apps.server.capserver init --dir STATE ; python3 -m apps.server.capserver serve --dir STATE --port 8788"""
from __future__ import annotations
import argparse, base64, hashlib, hmac, json, re, secrets, shutil, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.cli import Cli
from packages.capbuild import keygen, write_trust, load_private
from training.service import BuildService
from training.learning_package.examples import package_from_examples
from training.learning_package.schema import InvalidLearningPackage
from apps.desktop.app import NullOracle

MAX_BODY = 20 * 1024 * 1024
MAX_ROWS, MAX_DIM = 20000, 256
PKG_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,120}\.cap$")


def init(d: Path, signer="server-build-1"):
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "keys" / f"{signer}.private").exists():
        keygen(signer, d / "keys")
    write_trust(d / "trust.json", {signer: (d / "keys" / f"{signer}.public").read_text()})
    tok = d / "server.token"
    if not tok.exists():
        tok.write_text(secrets.token_hex(32)); tok.chmod(0o600)
    (d / "packages").mkdir(exist_ok=True)


class CapServer:
    def __init__(self, d: Path, signer="server-build-1"):
        self.d, self.signer = Path(d), signer
        self.key = load_private(signer, self.d / "keys")
        self.cli = Cli(self.d / "runtime", self.d / "trust.json")            # the server's own runtime (full device profile) for remote execution
        self.svc = BuildService(self.d / "keys", signer, self.d / "build", NullOracle())
        self.lock = threading.Lock()
        self.state_file = self.d / "catalog_state.json"
        self.state = json.loads(self.state_file.read_text()) if self.state_file.exists() else {"sequence": 0, "revoked": []}
        self.packages: dict[str, dict] = {}
        for f in sorted((self.d / "packages").glob("*.cap")):
            self._index(f)

    def _index(self, f: Path):
        import zipfile
        m = json.loads(zipfile.ZipFile(f).read("manifest.json"))
        self.packages[m["package_id"]] = {"package_id": m["package_id"], "capability_id": m["capability_id"], "version": m["version"], "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                                          "size": f.stat().st_size, "model_bytes": m["resources"]["model_bytes"], "params": m["resources"]["params"], "requires": m["requires"], "input_dim": m["io"]["input"]["dim"]}

    def register_package(self, path: Path):
        """Publish an already-built, signed package (also used by tests/experiments to publish hand-built packages)."""
        import zipfile
        pid = json.loads(zipfile.ZipFile(path).read("manifest.json"))["package_id"]
        dst = self.d / "packages" / f"{pid}.cap"                    # stored (and served) under its package id
        if Path(path).resolve() != dst.resolve():
            shutil.copy(path, dst)
        self._index(dst)
        rep = self.cli.import_caps(str(dst))[0]
        self._bump()
        return rep

    # ---- endorsement of device-learned packages ----------------------------------------------------------------------
    def enroll_device(self, key_id: str, public_b64: str):
        """Operator action (no HTTP endpoint): trust this device key *for endorsement only*."""
        f = self.d / "devices.json"; cur = json.loads(f.read_text()) if f.exists() else {}
        cur[key_id] = public_b64; f.write_text(json.dumps(cur))

    def endorse(self, package: bytes, examples: dict, min_acc: float = 0.9) -> dict:
        """Verify a device-signed package with the core (device trust roots only), evaluate it on the supplied examples, and, if it passes, re-sign it
        with the server key so other devices (which trust the server, not this device) can install it. Endorsement certifies integrity and
        consistency with the examples; it does not prove that the examples' labels are true."""
        import tempfile, zipfile, io
        devs = self.d / "devices.json"
        if not devs.exists() or not json.loads(devs.read_text()):
            return {"endorsed": False, "reason": "no devices are enrolled"}
        with self.lock, tempfile.TemporaryDirectory() as td:
            td = Path(td); (td / "in.cap").write_bytes(package); (td / "trust.json").write_text(devs.read_text())
            sandbox = Cli(td / "rt", td / "trust.json", "PC_FULL")
            rep = sandbox.import_caps(str(td / "in.cap"))[0]
            if not rep["activated"]:
                return {"endorsed": False, "reason": f"core rejected the package at step '{rep['failed_step']}'", "steps": [s["step"] + (":ok" if s["ok"] else ":FAILED") for s in rep["steps"]]}
            cap, ver = rep["capability_id"], rep["version"]
            xs, ys = examples["x"], examples["y"]
            cases = td / "cases.jsonl"; cases.write_text("\n".join(json.dumps({"intent": "x", "input": x, "expected_index": int(y)}) for x, y in zip(xs, ys)) + "\n")
            summ = sandbox._run_detect("keyword", "batch", "--cases", str(cases), capability=cap)["summary"]
            acc = summ["accuracy"]
            if not (acc >= min_acc):
                return {"endorsed": False, "reason": f"accuracy {acc:.3f} on the supplied examples < {min_acc}", "accuracy": acc}
            z = zipfile.ZipFile(io.BytesIO(package)); m = json.loads(z.read("manifest.json"))
            minor_patch = lambda v: tuple(map(int, v.split(".")))
            for p in self.packages.values():
                if p["capability_id"] == cap and minor_patch(p["version"]) >= minor_patch(ver):
                    return {"endorsed": False, "reason": f"server already has {cap} {p['version']}; a device-learned version must be newer than {p['version']}"}
            from packages.capbuild import build_cap
            tests = [(json.loads(l)["input"], json.loads(l)["expected"]) for l in z.read("tests/cases.jsonl").decode().splitlines() if l.strip()]
            hints = json.loads(z.read("routing/hints.json"))
            prov = {**m.get("provenance", {}), "endorsed_by": self.signer, "device_signer": m["signer"]["key_id"], "device_package_id": m["package_id"],
                    "endorsement_evaluation": {"examples": len(xs), "accuracy": acc, "min_acc": min_acc, "note": "integrity and consistency with the examples; label truth not verified"}}
            out = self.d / "build" / f"{cap}-{ver}-endorsed.cap"; out.parent.mkdir(exist_ok=True)
            build_cap(out, capability_id=cap, version=ver, model_bytes=z.read("model/model.onnx"), params=m["resources"]["params"], input_dim=m["io"]["input"]["dim"], labels=m["io"]["output"]["labels"],
                      tests=tests, keywords=hints["keywords"], description=hints["description"], signer_id=self.signer, signer_key=self.key, provenance=prov, min_accuracy=m["tests"]["min_accuracy"],
                      device_caps=m["requires"]["device_caps"], dependencies=m["dependencies"], input_stats=m.get("input_stats"), calibration=m.get("calibration"))
            r = self.register_package(out)
            return {"endorsed": r["activated"], "package_id": r["package_id"], "capability_id": cap, "version": ver, "accuracy": acc, "reason": None if r["activated"] else f"server import failed at {r.get('failed_step')}"}

    def _bump(self):
        self.state["sequence"] += 1; self.state_file.write_text(json.dumps(self.state))

    def catalog(self) -> dict:
        """Signed catalog: `body` is the exact string that was signed (no canonicalisation ambiguity)."""
        body = json.dumps({"sequence": self.state["sequence"], "generated_at": time.time(), "packages": sorted(self.packages.values(), key=lambda p: (p["capability_id"], p["version"])),
                           "revoked_keys": self.state["revoked"]}, sort_keys=True)
        sig = self.key.sign(body.encode())
        return {"body": body, "signature": {"alg": "ed25519", "key_id": self.signer, "sig_b64": base64.b64encode(sig).decode()}}

    def train(self, spec: dict) -> dict:
        cap = spec["capability_id"]
        with self.lock:
            minor = 0
            for p in self.packages.values():
                if p["capability_id"] == cap:
                    minor = max(minor, int(p["version"].split(".")[1]))
            labels = spec.get("labels") or [str(i) for i in range(max(spec["y"]) + 1)]
            kws = spec.get("keywords") or sorted({w for w in (spec.get("description", "") + " " + cap.replace("_", " ")).lower().split() if len(w) > 2})
            try:
                lp, held, info = package_from_examples(cap, spec.get("description", cap), kws, labels, spec["x"], spec["y"], seed=int(spec.get("seed", 0)))
                b = self.svc.build(lp, f"0.{minor + 1}.0", {p["capability_id"] for p in self.packages.values()}, heldout=held, standardize=True, min_acc=float(spec.get("min_acc", 0.9)),
                                   attempts=((32, 32), (64, 64), (128, 64)))
            except (ValueError, InvalidLearningPackage) as e:
                return {"learned": False, "reason": f"invalid examples: {str(e)[:160]}"}
            if not b.promoted:
                return {"learned": False, "reason": b.reason, "data": info}
            rep = self.register_package(b.cap_path)
            return {"learned": rep["activated"], "package_id": rep["package_id"], "version": f"0.{minor + 1}.0", "data": info, "heldout_accuracy": b.report["attempts"][-1]["heldout_accuracy"],
                    "failed_step": rep.get("failed_step")}

    def solve(self, intent: str, x: list[float]) -> dict:
        with self.lock:
            out = self.cli.solve(intent, x)
        out["executed_on"] = "server"
        return out


def make_handler(app: CapServer, token: str, port: int):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, code, obj=None, raw: bytes | None = None, ctype="application/json"):
            data = raw if raw is not None else json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(data)
        def _guard(self, auth=True):
            if (self.headers.get("Host") or "").lower() not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                self._send(403, {"error": "bad host header"}); return False
            if auth and not hmac.compare_digest(((self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()).encode(), token.encode()):
                self._send(401, {"error": "unauthorized"}); return False
            return True
        def do_GET(self):
            if self.path == "/health":
                if self._guard(False): self._send(200, {"ok": True})
                return
            if not self._guard(): return
            if self.path == "/catalog":
                with app.lock: self._send(200, app.catalog())
            elif self.path.startswith("/packages/"):
                name = self.path[len("/packages/"):]
                f = app.d / "packages" / name
                if not PKG_RE.match(name) or not f.is_file(): self._send(404, {"error": "not found"})
                else: self._send(200, raw=f.read_bytes(), ctype="application/octet-stream")
            else:
                self._send(404, {"error": "not found"})
        def do_POST(self):
            if not self._guard(): return
            if self.path not in ("/train", "/solve", "/endorse"): self._send(404, {"error": "not found"}); return
            try: n = int(self.headers.get("Content-Length") or 0)
            except ValueError: self._send(400, {"error": "bad content-length"}); return
            if n <= 0 or n > MAX_BODY: self._send(413 if n > MAX_BODY else 400, {"error": "body size not acceptable"}); return
            try:
                body = json.loads(self.rfile.read(n)); assert isinstance(body, dict)
            except Exception:
                self._send(400, {"error": "malformed JSON"}); return
            try:
                if self.path == "/endorse":
                    pkg = base64.b64decode(body["package_b64"], validate=True)
                    ex = body["examples"]
                    if not (isinstance(ex.get("x"), list) and isinstance(ex.get("y"), list) and len(ex["x"]) == len(ex["y"]) and 0 < len(ex["x"]) <= MAX_ROWS): raise ValueError("invalid examples")
                    self._send(200, app.endorse(pkg, ex, float(body.get("min_acc", 0.9))))
                elif self.path == "/solve":
                    x = body["input"]
                    if not isinstance(body.get("intent"), str) or not isinstance(x, list) or not all(isinstance(v, (int, float)) for v in x) or len(x) > 4096: raise ValueError("intent (string) and input (numbers) required")
                    self._send(200, app.solve(body["intent"], [float(v) for v in x]))
                else:
                    cap, xs, ys = body["capability_id"], body["x"], body["y"]
                    if not isinstance(cap, str) or not re.fullmatch(r"[A-Za-z0-9_]{1,60}", cap): raise ValueError("invalid capability_id")
                    if not (isinstance(xs, list) and isinstance(ys, list) and len(xs) == len(ys) and 0 < len(xs) <= MAX_ROWS and all(isinstance(r, list) and 0 < len(r) <= MAX_DIM for r in xs)): raise ValueError("invalid example arrays")
                    self._send(200, app.train(body))
            except (KeyError, ValueError, TypeError) as e:
                self._send(400, {"error": str(e)[:200]})
            except Exception as e:
                self._send(500, {"error": f"internal error ({type(e).__name__})"})
    return H


def serve(app: CapServer, port=0, token=None) -> HTTPServer:
    srv = HTTPServer(("127.0.0.1", port), None)
    srv.RequestHandlerClass = make_handler(app, token or (app.d / "server.token").read_text().strip(), srv.server_address[1])
    return srv


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("init", "serve"):
        p = sub.add_parser(n); p.add_argument("--dir", required=True)
    sub.choices["serve"].add_argument("--port", type=int, default=8788)
    a = ap.parse_args()
    if a.cmd == "init": init(Path(a.dir)); print("initialised", a.dir)
    else:
        app = CapServer(Path(a.dir)); s = serve(app, a.port); print(f"listening on http://127.0.0.1:{s.server_address[1]}"); s.serve_forever()


if __name__ == "__main__":
    main()
