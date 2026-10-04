"""Phase 12 gates: on-device learning (Rust trainer, ONNX writer, native packaging and signing), adaptation gates, device keys, server endorsement."""
import base64, hashlib, io, json, threading, zipfile
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort, pytest
from onnx import numpy_helper
from sklearn.datasets import load_iris, load_wine
from conftest import ROOT
from scripts.phase12_lab import Device
from scripts.cli import Cli
from apps.server.capserver import init, CapServer, serve
from packages.capbuild import keygen, write_trust

IRIS_INTENT = "classify iris flower species from measurements"


def iris_spec(**over):
    d = load_iris()
    return {"capability_id": "iris_species", "description": IRIS_INTENT, "keywords": ["iris", "species", "flower", "classify", "measurements"], "labels": [str(t) for t in d.target_names],
            "x": d.data.tolist(), "y": d.target.tolist(), "min_accuracy": 0.9, **over}


@pytest.fixture
def trust(tmp_path):
    keygen("server-build-1", tmp_path / "srvkeys"); write_trust(tmp_path / "trust.json", {"server-build-1": (tmp_path / "srvkeys/server-build-1.public").read_text()})
    return tmp_path / "trust.json"


@pytest.fixture
def dev(tmp_path, trust):
    return Device(tmp_path / "dev", trust)


def model_of(cap: Path):
    return zipfile.ZipFile(cap).read("model/model.onnx")


def manifest_of(cap: Path):
    return json.loads(zipfile.ZipFile(cap).read("manifest.json"))


def test_learning_on_the_device_installs_a_signed_capability_through_the_normal_gates(dev):
    r = dev.learn(iris_spec())
    assert r["learned"] and r["test_accuracy"] >= 0.9 and r["import"]["activated"]
    assert [s["step"] for s in r["import"]["steps"]] == ["parse", "signature", "hashes", "compatibility", "sandbox_load", "bundled_tests", "resource_check", "activate"]
    out = dev.cli.solve(IRIS_INTENT, load_iris().data[0].tolist())
    assert out["result"] == "ANSWER" and out["label"] == "setosa"
    m = manifest_of(dev.active_package("iris_species"))
    assert m["signer"]["key_id"] == "device-1" and m["provenance"]["strategy"] == "new_module" and m["provenance"]["training"]["trainer"] == "aicore-train-lite"


def test_the_exported_onnx_is_valid_for_external_tools_and_agrees_with_the_runtime(dev):
    dev.learn(iris_spec())
    cap = dev.active_package("iris_species"); data = model_of(cap)
    onnx.checker.check_model(onnx.load_from_string(data))                                  # an independent implementation accepts the graph
    sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"]); T = manifest_of(cap)["calibration"]["temperature"]; assert T >= 1.0          # calibration never sharpens
    T = 1.0                                                                                  # the default detection mode does not apply calibration
    rows = load_iris().data[::7]
    f = dev.root.parent / "rows.jsonl"; f.write_text("\n".join(json.dumps({"intent": IRIS_INTENT, "input": r.tolist(), "expected_index": 0}) for r in rows) + "\n")
    res = dev.cli._run_detect("keyword", "batch", "--cases", str(f), capability="iris_species")["results"]
    for r, row in zip(res, rows):
        logits = sess.run(None, {"input": row.astype(np.float32)[None]})[0][0] / T
        p = np.exp(logits - logits.max()); p /= p.sum()
        assert np.abs(p - np.array(r["probs"])).max() < 1e-3          # the CLI rounds probabilities to 3 decimals


def test_device_signed_packages_are_trusted_only_by_the_device_that_holds_the_key(dev, tmp_path, trust):
    dev.learn(iris_spec())
    cap = dev.active_package("iris_species")
    other = Cli(tmp_path / "other", trust)                                                  # a different runtime: trusts only the server key
    rep = other.import_caps(str(cap))[0]
    assert rep["activated"] is False and rep["failed_step"] == "signature"
    same_key_other_dir = Device(tmp_path / "dev2", trust, with_key=False)                   # without the device key the package still fails
    assert same_key_other_dir.cli.import_caps(str(cap))[0]["failed_step"] == "signature"


def test_training_is_deterministic_for_a_seed(tmp_path, trust):
    h = []
    for i in range(2):
        d = Device(tmp_path / f"d{i}", trust); d.learn(iris_spec(seed=3)); h.append(hashlib.sha256(model_of(d.active_package("iris_species"))).hexdigest())
    d = Device(tmp_path / "d9", trust); d.learn(iris_spec(seed=4)); h.append(hashlib.sha256(model_of(d.active_package("iris_species"))).hexdigest())
    assert h[0] == h[1] and h[0] != h[2]


def test_learning_without_a_device_key_fails_cleanly(tmp_path, trust):
    d = Device(tmp_path / "nokey", trust, with_key=False)
    with pytest.raises(RuntimeError, match="no device signing key"):
        d.learn(iris_spec(), with_key=False)
    assert d.cli.list() == []


@pytest.mark.parametrize("mut,msg", [
    (lambda s: {**s, "x": s["x"][:10], "y": s["y"][:10]}, "at least 30"),
    (lambda s: {**s, "y": [5] + s["y"][1:]}, "out of range"),
    (lambda s: {**s, "capability_id": "bad id!"}, "capability_id"),
    (lambda s: {**s, "labels": ["only"]}, "at least 2 labels"),
    (lambda s: {**s, "x": [[float("nan")] * 4] + s["x"][1:]}, ""),
])
def test_invalid_training_requests_are_refused(dev, mut, msg):
    try:
        r = dev.learn(mut(iris_spec())); assert r["learned"] is False
    except (RuntimeError, ValueError) as e:
        assert msg in str(e)
    assert dev.cli.list() == []


def test_a_class_with_too_few_examples_is_refused(dev):
    s = iris_spec(); keep = [i for i, y in enumerate(s["y"]) if y != 2 or i == 149 or i == 148]
    s["x"], s["y"] = [s["x"][i] for i in keep], [s["y"][i] for i in keep]
    with pytest.raises(RuntimeError, match="at least 3 per class"): dev.learn(s)
    assert dev.cli.list() == []


def test_noise_is_rejected_by_the_accuracy_gate_not_installed(dev):
    rng = np.random.default_rng(0)
    r = dev.learn({"capability_id": "noise", "labels": ["a", "b"], "x": rng.uniform(size=(200, 5)).tolist(), "y": rng.integers(0, 2, 200).tolist(), "min_accuracy": 0.9})
    assert r["learned"] is False and "held-out accuracy" in r["reason"] and len(r["attempts"]) == 3 and dev.cli.list() == []


def initializers(cap: Path):
    m = onnx.load_from_string(model_of(cap))
    return {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}


def drift_data(shift=1.0, seed=0):
    d = load_iris(); rng = np.random.default_rng(seed)
    idx = rng.permutation(len(d.data)); x = d.data[idx].copy(); x[:, 2] += shift * 1.5      # a miscalibrated sensor: petal length reads 1.5 cm high (base model falls to ~0.6)
    return x, d.target[idx]


def adapt_setup(dev):
    dev.learn(iris_spec(seed=1))
    x, y = drift_data()
    d = load_iris()
    return {"capability_id": "iris_species", "x": x[:30].tolist(), "y": y[:30].tolist(), "x_test": x[60:].tolist(), "y_test": y[60:].tolist(),
            "x_old": d.data[::3].tolist(), "y_old": d.target[::3].tolist(), "seed": 1}


def test_head_only_adaptation_freezes_the_trunk_and_full_adaptation_does_not(tmp_path, trust):
    out = {}
    for mode in ("head", "full"):
        dv = Device(tmp_path / mode, trust); spec = adapt_setup(dv); before = initializers(dv.active_package("iris_species"))
        r = dv.adapt({**spec, "mode": mode, "max_old_drop": 1.0, "min_gain": -1.0})      # gates disabled here: this test is about what changes
        assert r["learned"], r
        after = initializers(dv.active_package("iris_species")); out[mode] = (before, after, r)
        assert manifest_of(dv.active_package("iris_species"))["version"] == "0.1.1"
    b, a, _ = out["head"]
    assert np.array_equal(b["w0"], a["w0"]) and np.array_equal(b["w1"], a["w1"]) and np.array_equal(b["0.mean"], a["0.mean"]) and not np.array_equal(b["w2"], a["w2"])
    b, a, _ = out["full"]
    assert not np.array_equal(b["w0"], a["w0"]) and np.array_equal(b["0.mean"], a["0.mean"])        # standardisation constants stay fixed in both modes
    assert out["head"][2]["after"]["trainable"] < out["full"][2]["after"]["trainable"]


def test_adaptation_improves_on_the_drifted_data(dev):
    spec = adapt_setup(dev)
    r = dev.adapt({**spec, "mode": "full", "max_old_drop": 1.0})
    assert r["learned"] and r["after"]["new_test_accuracy"] > r["before"]["new_test_accuracy"] + 0.05


def test_adaptation_that_would_wreck_the_old_behaviour_is_refused(dev):
    spec = adapt_setup(dev)
    wrong = {**spec, "y": [(y + 1) % 3 for y in spec["y"]], "y_test": [(y + 1) % 3 for y in spec["y_test"]], "mode": "full", "max_old_drop": 0.02}     # conflicting labels
    before = dev.cli.list()[0]["active_version"]
    r = dev.adapt(wrong)
    assert r["learned"] is False and "regression on old data" in r["reason"] and dev.cli.list()[0]["active_version"] == before


def test_adaptation_without_improvement_is_refused(dev):
    spec = adapt_setup(dev); d = load_iris()
    nochange = {**spec, "x": d.data[:30].tolist(), "y": d.target[:30].tolist(), "x_test": d.data[60:].tolist(), "y_test": d.target[60:].tolist(), "mode": "head", "min_gain": 0.05}
    r = dev.adapt(nochange)
    assert r["learned"] is False and "no improvement" in r["reason"]


def test_adaptation_works_on_a_capability_that_was_signed_by_the_server(tmp_path, trust):
    from scripts.phase11_lab import Lab
    L = Lab(tmp_path / "lab"); 
    try:
        d = load_iris(); L.client._http("POST", "/train", {"capability_id": "iris_species", "description": IRIS_INTENT, "labels": ["a", "b", "c"], "x": d.data.tolist(), "y": d.target.tolist(), "min_acc": 0.9})
        L.client.sync()
        dv = Device(tmp_path / "dev", L.dev_trust, with_key=True); dv.cli.import_caps(str(next((L.local.root / "store").glob("iris_species*.cap"))))
        x, y = drift_data()
        r = dv.adapt({"capability_id": "iris_species", "mode": "head", "x": x[:30].tolist(), "y": y[:30].tolist(), "x_test": x[60:].tolist(), "y_test": y[60:].tolist(), "x_old": d.data[::3].tolist(), "y_old": d.target[::3].tolist(), "max_old_drop": 1.0})
        assert r["learned"], r
        m = manifest_of(dv.active_package("iris_species")); assert m["signer"]["key_id"] == "device-1" and m["provenance"]["base_version"] == "0.1.0"
    finally:
        L.stop()


def test_router_update_on_the_device_keeps_the_model_and_bumps_the_version(dev):
    dev.learn(iris_spec())
    before = hashlib.sha256(model_of(dev.active_package("iris_species"))).hexdigest()
    assert dev.cli.solve("what sort of bloom is this", load_iris().data[0].tolist())["result"] == "NEEDS_HELP"
    rep = dev.alias("iris_species", ["bloom"])
    assert rep["activated"]
    assert hashlib.sha256(model_of(dev.active_package("iris_species"))).hexdigest() == before
    assert manifest_of(dev.active_package("iris_species"))["version"] == "0.1.1"
    assert dev.cli.solve("what sort of bloom is this", load_iris().data[0].tolist())["result"] == "ANSWER"


def test_device_key_files_are_private_and_distinct(tmp_path, trust):
    a, b = Device(tmp_path / "a", trust), Device(tmp_path / "b", trust)
    import stat
    assert stat.S_IMODE(a.key_file.stat().st_mode) == 0o600 and a.pub_b64 != b.pub_b64
    assert json.loads(a.key_file.read_text())["seed_hex"] not in a.pub_b64


# ------------------------------------------------------------------------------------------------ server endorsement
@pytest.fixture
def server(tmp_path):
    init(tmp_path / "srv"); app = CapServer(tmp_path / "srv"); s = serve(app); threading.Thread(target=s.serve_forever, daemon=True).start()
    yield app, s
    s.shutdown()


def post(srv, path, body):
    import urllib.request, urllib.error
    app, s = srv; tok = (app.d / "server.token").read_text().strip(); port = s.server_address[1]
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json", "Host": f"127.0.0.1:{port}"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r: return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e: return e.code, json.loads(e.read())


def test_server_endorses_a_device_learned_package_and_a_pc_that_trusts_only_the_server_can_install_it(tmp_path, server):
    app, s = server
    trust = tmp_path / "devtrust.json"; trust.write_text((app.d / "trust.json").read_text())
    dev = Device(tmp_path / "dev", trust); dev.learn(iris_spec())
    cap = dev.active_package("iris_species"); d = load_iris()
    status, out = post(server, "/endorse", {"package_b64": base64.b64encode(cap.read_bytes()).decode(), "examples": {"x": d.data.tolist(), "y": d.target.tolist()}})
    assert out["endorsed"] is False and "no devices are enrolled" in out["reason"]
    app.enroll_device("device-1", dev.pub_b64)
    status, out = post(server, "/endorse", {"package_b64": base64.b64encode(cap.read_bytes()).decode(), "examples": {"x": d.data.tolist(), "y": d.target.tolist()}, "min_acc": 0.9})
    assert out["endorsed"] is True and out["accuracy"] >= 0.9, out
    pc = Cli(tmp_path / "pc", trust)                                                      # trusts only the server key
    from apps.hybrid.client import HybridClient
    hc = HybridClient(pc, f"http://127.0.0.1:{s.server_address[1]}", (app.d / "server.token").read_text().strip(), tmp_path / "pcwork", trust, 8 << 20)
    rep = hc.sync(); assert [i[0] for i in rep.installed] == ["iris_species"]
    assert pc.solve(IRIS_INTENT, d.data[0].tolist())["label"] == "setosa"
    m = manifest_of(next((tmp_path / "pc/store").glob("iris_species*.cap")))
    assert m["signer"]["key_id"] == "server-build-1" and m["provenance"]["endorsed_by"] == "server-build-1" and m["provenance"]["device_signer"] == "device-1"


def test_endorsement_rejects_unenrolled_tampered_and_inaccurate_packages(tmp_path, server):
    app, s = server
    trust = tmp_path / "devtrust.json"; trust.write_text((app.d / "trust.json").read_text())
    dev = Device(tmp_path / "dev", trust); dev.learn(iris_spec()); cap = dev.active_package("iris_species"); d = load_iris()
    ex = {"x": d.data.tolist(), "y": d.target.tolist()}
    other = Device(tmp_path / "other", trust); app.enroll_device("someone-else", other.pub_b64)          # a different device is enrolled, ours is not
    assert post(server, "/endorse", {"package_b64": base64.b64encode(cap.read_bytes()).decode(), "examples": ex})[1]["reason"].startswith("core rejected the package at step 'signature'")
    app.enroll_device("device-1", dev.pub_b64)
    bad = bytearray(cap.read_bytes()); z = zipfile.ZipFile(io.BytesIO(bytes(bad)))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zo:
        for n in z.namelist(): zo.writestr(n, (b"\x00" + z.read(n)[1:]) if n == "model/model.onnx" else z.read(n))
    assert "hashes" in post(server, "/endorse", {"package_b64": base64.b64encode(buf.getvalue()).decode(), "examples": ex})[1]["reason"]
    wrong = {"x": ex["x"], "y": [(y + 1) % 3 for y in ex["y"]]}                                              # examples the model contradicts
    assert "accuracy" in post(server, "/endorse", {"package_b64": base64.b64encode(cap.read_bytes()).decode(), "examples": wrong})[1]["reason"]
    assert post(server, "/endorse", {"package_b64": "!!notbase64", "examples": ex})[0] == 400
    assert app.packages == {}                                                                                  # nothing was published


def test_head_only_adaptation_cannot_repair_an_input_shift_and_the_gate_says_so(dev):
    """Negative result kept as a test: a frozen trunk cannot undo a shifted input feature, so the default gate refuses the update."""
    spec = adapt_setup(dev)
    r = dev.adapt({**spec, "mode": "head", "max_old_drop": 1.0})
    assert r["learned"] is False and "no improvement" in r["reason"]
    assert dev.cli.list()[0]["active_version"] == "0.1.0"
