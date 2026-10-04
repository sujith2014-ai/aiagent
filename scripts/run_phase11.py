"""Phase 11: hybrid local/server execution, signed delivery, offline mode: measurements on loopback (no real network: latencies are lower bounds)."""
from __future__ import annotations
import base64, json, shutil, sys, tempfile, time, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits, load_breast_cancer
from scripts.phase11_lab import Lab
from scripts.cli import ROOT
from apps.hybrid.client import DeliveryError, placement, LOCAL, SERVER, QUEUE, REFUSE
from packages.capbuild import keygen, load_private

RUN = ROOT / "runs" / "phase11"
TASKS = [("iris_species", load_iris, "classify iris flower species from sepal and petal measurements"), ("wine_cultivar", load_wine, "identify wine cultivar from chemical analysis"),
         ("digit_recognition", load_digits, "recognize handwritten digit from 8x8 pixel intensities"), ("tumor_malignancy", load_breast_cancer, "classify breast tumor as malignant or benign from cell measurements")]


def pct(a, q): a = sorted(a); return a[min(len(a) - 1, int(q * (len(a) - 1)))]


def main():
    if RUN.exists(): shutil.rmtree(RUN)
    RUN.mkdir(parents=True)
    rep = {"benchmark_format": "bench/1", "experiment": "phase11-hybrid-delivery-offline", "platform": platform.platform(), "date": time.strftime("%Y-%m-%d"),
           "network": "loopback only: latencies are lower bounds; no TLS; no real WAN"}
    L = Lab(RUN / "lab", device="PC_CONSTRAINED", max_model_bytes=8 << 20)
    # ---- A. server-side training and signed delivery for the four real tasks ----
    deliveries = {}
    for cap, loader, intent in TASKS:
        d = loader(); spec = {"capability_id": cap, "description": intent, "labels": [str(t) for t in d.target_names], "x": d.data.tolist(), "y": d.target.tolist(), "min_acc": 0.9}
        upload_bytes = len(json.dumps(spec))
        t0 = time.time(); r = L.client._http("POST", "/train", spec); t_train = time.time() - t0
        t0 = time.time(); sr = L.client.sync(); t_sync = time.time() - t0
        pkg = next(p for p in L.app.packages.values() if p["capability_id"] == cap)
        deliveries[cap] = {"learned": r["learned"], "server_train_and_sign_s": round(t_train, 2), "sync_verify_install_s": round(t_sync, 3), "upload_bytes_json": upload_bytes, "package_bytes": pkg["size"],
                           "installed_on_device": [i[0] for i in sr.installed] == [cap], "heldout_accuracy": r.get("heldout_accuracy")}
    rep["delivery"] = deliveries
    rep["catalog_bytes"] = len(L.client._http("GET", "/catalog", raw=True))
    rep["device_holds_private_key"] = bool(list((RUN / "lab/dev").rglob("*.private")))
    # ---- B. local vs remote execution latency ----
    d = load_iris(); row = d.data[0].tolist(); n = 300
    loc, rem = [], []
    for _ in range(n):
        t = time.perf_counter(); L.local.solve("classify iris flower species from sepal and petal measurements", row); loc.append((time.perf_counter() - t) * 1e6)
    for _ in range(n):
        t = time.perf_counter(); L.client._http("POST", "/solve", {"intent": "classify iris flower species from sepal and petal measurements", "input": row}); rem.append((time.perf_counter() - t) * 1e6)
    rep["latency_us"] = {"local_incl_process_spawn": {"p50": pct(loc, .5), "p95": pct(loc, .95)}, "remote_loopback_http": {"p50": pct(rem, .5), "p95": pct(rem, .95)},
                         "note": "local numbers include spawning the CLI process per call (the harness), so they overstate on-device latency; core inference alone is ~1-4 us (benchmarks/reports/backend_bench.json)"}
    # ---- C. placement outcomes ----
    sc = {}
    sc["known_task"] = {k: v for k, v in L.client.solve("classify iris flower species from sepal and petal measurements", row).items() if k in ("placement", "executed_on", "result")}
    sc["unknown_task_online"] = {k: v for k, v in L.client.solve("translate this sentence", [0.1, 0.2]).items() if k in ("placement", "executed_on", "result")}
    before = L.client.requests_to_server
    sc["sensitive_unknown_task"] = {**{k: v for k, v in L.client.solve("translate my diary", [0.1, 0.2], sensitive=True).items() if k in ("placement", "result")}, "server_requests_made": L.client.requests_to_server - before}
    sc["sensitive_examples_upload"] = {**L.client.teach({"capability_id": "secret", "x": [[1.0]], "y": [0]}, sensitive=True, consent_to_upload=True), "server_requests_made": L.client.requests_to_server - before}
    rep["placement"] = sc
    # ---- D. offline timeline ----
    port = L.port; L.stop(); tl = []
    for intent, x, kind in [("classify iris flower species from sepal and petal measurements", row, "known"), ("identify wine cultivar from chemical analysis", load_wine().data[0].tolist(), "known"),
                            ("recognize handwritten digit from 8x8 pixel intensities", load_digits().data[3].tolist(), "known"), ("book a flight", [0.1], "unknown"), ("summarise this", [0.2, 0.3], "unknown")]:
        o = L.client.solve(intent, x); tl.append({"kind": kind, "placement": o["placement"], "result": o["result"]})
    t = L.client.teach({"capability_id": "queued_task", "description": "queued", "labels": ["a", "b"], "x": np.random.default_rng(0).uniform(size=(80, 3)).tolist(), "y": ([0, 1] * 40), "min_acc": 0.5}, consent_to_upload=True)
    pend = len(L.client.pending())
    f1 = L.client.flush()
    L.start(port=port); f2 = L.client.flush()
    rep["offline"] = {"timeline": tl, "known_answered_offline": sum(1 for x in tl if x["kind"] == "known" and x["result"] == "ANSWER"), "known_total": sum(1 for x in tl if x["kind"] == "known"),
                      "unknown_queued": sum(1 for x in tl if x["placement"] == QUEUE), "teach_while_offline": {k: t.get(k) for k in ("uploaded", "queued")}, "pending_before_reconnect": pend,
                      "flush_while_offline": {k: f1.get(k) for k in ("flushed", "still_queued")}, "flush_after_reconnect": {k: f2.get(k) for k in ("flushed", "still_queued")}}
    # ---- E. attacks on the delivery path ----
    atk = {}
    def attempt(name, setup, expect):
        fresh = Lab(RUN / f"atk_{name}", device="PC_CONSTRAINED")
        try:
            iris = load_iris(); fresh.client._http("POST", "/train", {"capability_id": "iris_species", "description": "classify iris", "labels": ["a", "b", "c"], "x": iris.data.tolist(), "y": iris.target.tolist(), "min_acc": 0.9})
            out = setup(fresh)
            atk[name] = {"outcome": out, "device_capabilities_after": [c["capability_id"] for c in fresh.local.list()], "defended": expect(out, fresh)}
        finally:
            fresh.stop()
    def patch(fresh, fn):
        orig = fresh.client._http; fresh.client._http = lambda method, path, body=None, raw=False, max_bytes=64 << 20: fn(orig, method, path, body, raw, max_bytes)
    def run_sync(fresh):
        try: r = fresh.client.sync(); return {"installed": r.installed, "rejected": [x[1] for x in r.rejected], "warnings": r.warnings}
        except DeliveryError as e: return {"refused": str(e)}
    def tamper_catalog(f):
        patch(f, lambda o, m, p, b, r, mb: (lambda x: {**x, "body": x["body"].replace('"sequence": ', '"sequence": 9')} if p == "/catalog" else x)(o(m, p, b, r, mb))); return run_sync(f)
    attempt("tampered_catalog", tamper_catalog, lambda o, f: "signature invalid" in o.get("refused", ""))
    def rogue_catalog(f):
        keygen("evil", f.tmp / "ek"); ev = load_private("evil", f.tmp / "ek")
        patch(f, lambda o, m, p, b, r, mb: (lambda x: {**x, "signature": {"alg": "ed25519", "key_id": "evil", "sig_b64": base64.b64encode(ev.sign(x["body"].encode())).decode()}} if p == "/catalog" else x)(o(m, p, b, r, mb))); return run_sync(f)
    attempt("catalog_signed_by_unknown_key", rogue_catalog, lambda o, f: "untrusted or revoked" in o.get("refused", ""))
    def tamper_pkg(f):
        def m(o, meth, p, b, r, mb):
            x = o(meth, p, b, r, mb)
            if p.startswith("/packages/"): y = bytearray(x); y[len(y) // 2] ^= 0xFF; return bytes(y)
            return x
        patch(f, m); return run_sync(f)
    attempt("modified_package_in_transit", tamper_pkg, lambda o, f: not o.get("installed") and any("do not match" in r for r in o.get("rejected", [])))
    def replay(f):
        old = f.client._http("GET", "/catalog"); f.client.sync()
        f.client._http("POST", "/train", {"capability_id": "iris_species", "description": "classify iris", "labels": ["a", "b", "c"], "x": load_iris().data.tolist(), "y": load_iris().target.tolist(), "min_acc": 0.9, "seed": 1}); f.client.sync()
        patch(f, lambda o, m, p, b, r, mb: old if p == "/catalog" else o(m, p, b, r, mb)); return run_sync(f)
    attempt("catalog_replay", replay, lambda o, f: "replay" in o.get("refused", ""))
    def freeze(f):
        f.client.max_age = 0.0; time.sleep(0.05); return run_sync(f)
    attempt("stale_catalog_freeze", freeze, lambda o, f: "maximum allowed age" in o.get("refused", ""))
    def downgrade(f):
        f.client.sync(); f.client._http("POST", "/train", {"capability_id": "iris_species", "description": "classify iris", "labels": ["a", "b", "c"], "x": load_iris().data.tolist(), "y": load_iris().target.tolist(), "min_acc": 0.9, "seed": 1}); f.client.sync()
        old = [p for p in f.app.packages.values() if p["version"] == "0.1.0"][0]; f.app.packages = {old["package_id"]: old}; f.app._bump(); return run_sync(f)
    attempt("downgrade_via_signed_catalog", downgrade, lambda o, f: not o["installed"] and any("possible downgrade" in w for w in o["warnings"]) and f.local.list()[0]["active_version"] == "0.2.0")
    rep["attacks"] = atk
    out = ROOT / "benchmarks/reports/phase11.json"; out.write_text(json.dumps(rep, indent=1, default=str)); L.stop()
    print(json.dumps({"delivery": {k: (v["package_bytes"], v["server_train_and_sign_s"], v["sync_verify_install_s"]) for k, v in deliveries.items()}, "latency": rep["latency_us"]["remote_loopback_http"], "placement": sc,
                      "offline": {k: v for k, v in rep["offline"].items() if k != "timeline"}, "attacks": {k: v["defended"] for k, v in atk.items()}}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
