"""Phase 9 gates: desktop service (auth, hardening, persistence) and learning real datasets from user-supplied examples."""
import json, os, stat, threading, urllib.request, urllib.error
from pathlib import Path
import pytest
from conftest import ROOT
from apps.desktop.app import DesktopApp, init, serve
from sklearn.datasets import load_iris, load_wine


@pytest.fixture
def desktop(tmp_path):
    cfg = init(tmp_path / "state")
    app = DesktopApp(cfg)
    srv = serve(app, port=0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    token = Path(app.cfg["api"]["token_file"]).read_text().strip()
    def call(method, path, body=None, headers=None, raw=None, host=None, auth=True):
        h = {"Host": host or f"127.0.0.1:{port}", **(headers or {})}
        if auth: h["Authorization"] = f"Bearer {token}"
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        if data is not None: h.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    yield app, call, cfg, srv
    srv.shutdown()


def iris_payload(**over):
    d = load_iris()
    return {"capability_id": "iris_species", "description": "classify iris flower species from sepal and petal measurements", "labels": [str(t) for t in d.target_names],
            "x": d.data.tolist(), "y": d.target.tolist(), "min_acc": 0.9, **over}


def test_init_creates_a_private_token_and_local_only_config(desktop):
    app, call, cfg, srv = desktop
    tok = Path(app.cfg["api"]["token_file"])
    assert stat.S_IMODE(tok.stat().st_mode) == 0o600 and len(tok.read_text().strip()) == 64
    assert app.cfg["api"]["host"] == "127.0.0.1"


def test_health_is_open_everything_else_needs_the_token(desktop):
    app, call, cfg, srv = desktop
    assert call("GET", "/health", auth=False)[0] == 200
    for path in ("/status", "/capabilities"):
        assert call("GET", path, auth=False)[0] == 401
    assert call("POST", "/solve", {"intent": "x", "input": [1]}, auth=False)[0] == 401
    code, _ = call("GET", "/status", headers={"Authorization": "Bearer wrong"}, auth=False)
    assert code == 401
    assert call("GET", "/status")[0] == 200


def test_dns_rebinding_and_browser_requests_are_refused(desktop):
    app, call, cfg, srv = desktop
    assert call("GET", "/status", host="evil.example.com")[0] == 403
    assert call("GET", "/status", headers={"Origin": "http://evil.example.com"})[0] == 403


@pytest.mark.parametrize("kwargs,code", [
    (dict(raw=b"{not json"), 400), (dict(raw=b"[1,2]"), 400), (dict(body={"intent": "x"}), 400), (dict(body={"intent": 5, "input": [1]}), 400),
    (dict(body={"intent": "x", "input": ["a"]}), 400), (dict(raw=b"{}", headers={"Content-Type": "text/plain"}), 415),
])
def test_malformed_requests_are_rejected_cleanly(desktop, kwargs, code):
    app, call, cfg, srv = desktop
    status, body = call("POST", "/solve", **kwargs)
    assert status == code and "error" in body and "Traceback" not in json.dumps(body)


def test_oversized_declared_body_is_refused_before_it_is_read(desktop):
    import http.client
    app, call, cfg, srv = desktop
    token = Path(app.cfg["api"]["token_file"]).read_text().strip()
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    conn.putrequest("POST", "/solve"); conn.putheader("Authorization", f"Bearer {token}"); conn.putheader("Content-Type", "application/json")
    conn.putheader("Content-Length", str(6 * 1024 * 1024)); conn.endheaders()          # declare 6 MB, send nothing
    r = conn.getresponse()
    assert r.status == 413 and "error" in json.loads(r.read())


def test_unknown_routes_and_methods(desktop):
    app, call, cfg, srv = desktop
    assert call("GET", "/nope")[0] == 404 and call("POST", "/nope", {})[0] == 404


def test_teach_by_example_then_solve_over_http_without_any_teacher(desktop):
    app, call, cfg, srv = desktop
    code, out = call("POST", "/teach", iris_payload())
    assert code == 200 and out["learned"] is True and out["teacher_used"] is False and out["heldout_accuracy"] >= 0.9
    d = load_iris()
    for row, truth in [(d.data[0], "setosa"), (d.data[75], "versicolor"), (d.data[140], "virginica")]:
        code, r = call("POST", "/solve", {"intent": "classify iris flower species from measurements", "input": row.tolist()})
        assert code == 200 and r["result"] == "ANSWER" and r["label"] == truth, r
    code, st = call("GET", "/status")
    assert [c["capability_id"] for c in st["capabilities"]] == ["iris_species"]


def test_out_of_range_measurements_are_refused_not_extrapolated(desktop):
    app, call, cfg, srv = desktop
    call("POST", "/teach", iris_payload())
    code, r = call("POST", "/solve", {"intent": "classify iris flower species from measurements", "input": [90.0, 80.0, 70.0, 60.0]})
    assert r["result"] == "NEEDS_HELP" and r["reason_code"] == "OUT_OF_DISTRIBUTION"


def test_state_survives_an_app_restart_and_reteaching_bumps_the_version(desktop, tmp_path):
    app, call, cfg, srv = desktop
    assert call("POST", "/teach", iris_payload())[1]["learned"]
    app2 = DesktopApp(cfg)                                                       # restart: new process state, same directory
    d = load_iris()
    assert app2.solve("classify iris flower species from measurements", d.data[0].tolist())["label"] == "setosa"
    out = app2.teach(iris_payload(seed=1))
    assert out["learned"] and out["version"] == "0.2.0"
    assert [c["active_version"] for c in app2.status()["capabilities"]] == ["0.2.0"]


def test_bad_examples_are_refused_without_installing_anything(desktop):
    app, call, cfg, srv = desktop
    code, out = call("POST", "/teach", {"capability_id": "junk", "x": [[1.0, 2.0]] * 30, "y": [0, 1] * 15, "min_acc": 0.9})
    assert out["learned"] is False and app.status()["capabilities"] == []        # contradictory duplicates collapse below the minimum sizes
    code, out = call("POST", "/teach", {"capability_id": "bad name!", "x": [[1.0]], "y": [0]})
    assert code == 400


def test_hard_task_is_rejected_by_the_gate_not_silently_installed(desktop):
    import random
    app, call, cfg, srv = desktop
    r = random.Random(0); x = [[r.random() for _ in range(6)] for _ in range(300)]; y = [r.randrange(2) for _ in range(300)]     # pure noise
    out = call("POST", "/teach", {"capability_id": "noise", "x": x, "y": y, "min_acc": 0.9})[1]
    assert out["learned"] is False and "heldout accuracy" in out["reason"] and app.status()["capabilities"] == []


def test_two_real_datasets_coexist_and_route_by_intent_and_shape(desktop):
    app, call, cfg, srv = desktop
    w = load_wine()
    assert call("POST", "/teach", iris_payload())[1]["learned"]
    assert call("POST", "/teach", {"capability_id": "wine_cultivar", "description": "identify wine cultivar from chemical analysis", "labels": [str(t) for t in w.target_names],
                                    "x": w.data.tolist(), "y": w.target.tolist(), "min_acc": 0.9})[1]["learned"]
    r = call("POST", "/solve", {"intent": "identify wine cultivar from chemical analysis", "input": w.data[0].tolist()})[1]
    assert r["capability_id"] == "wine_cultivar" and r["label"] == "class_0"
    r = call("POST", "/solve", {"intent": "classify iris flower species from measurements", "input": load_iris().data[0].tolist()})[1]
    assert r["capability_id"] == "iris_species"


def test_novelty_rules_strict_vs_balanced_on_real_iris(desktop):
    app, call, cfg, srv = desktop
    assert call("POST", "/teach", iris_payload())[1]["learned"]
    base = load_iris().data[0].tolist()                           # sepal length range in training is ~4.3-7.9
    one_moderate = [9.0] + base[1:]                               # one feature a little outside the range (z ~ 4)
    two_outside = [9.0, 5.5] + base[2:]
    extreme_single = [60.0] + base[1:]                            # one feature absurdly far (z >> 12)
    intent = "classify iris flower species from measurements"
    def run(rule, x):
        return app.cli._run_detect("full", "solve", "--intent", intent, "--input", ",".join(map(str, x)), novelty=rule)["result"]
    assert run("strict", base) == "ANSWER" and run("balanced", base) == "ANSWER"
    assert run("strict", one_moderate) == "NEEDS_HELP" and run("balanced", one_moderate) == "ANSWER"   # documented trade-off
    assert run("strict", two_outside) == "NEEDS_HELP" and run("balanced", two_outside) == "NEEDS_HELP"
    assert run("strict", extreme_single) == "NEEDS_HELP" and run("balanced", extreme_single) == "NEEDS_HELP"
