"""Phase 11 gates: hybrid placement, signed delivery, offline queue, privacy, attacks on the delivery path."""
import base64, itertools, json, socket, threading, time, urllib.request, urllib.error
from pathlib import Path
import numpy as np, pytest
from sklearn.datasets import load_iris
from conftest import ROOT
from apps.server.capserver import init, CapServer, serve
from apps.hybrid.client import HybridClient, DeliveryError, placement, LOCAL, SERVER, QUEUE, REFUSE
from scripts.cli import Cli
from packages.capbuild import keygen, load_private, build_cap, write_trust
from training.exporters.onnx_export import export_onnx
from training.trainer.train import make_mlp

from scripts.phase11_lab import Lab, iris_spec, IRIS_INTENT


@pytest.fixture
def lab(tmp_path):
    L = Lab(tmp_path)
    yield L
    try: L.stop()
    except Exception: pass


def teach_iris(L, **over):
    r = L.client.teach(iris_spec(**over), consent_to_upload=True)
    assert r["uploaded"] and r["learned"], r
    return r


# ----------------------------------------------------------------------------------------------- placement
@pytest.mark.parametrize("local,sensitive,online,allow", list(itertools.product(["ANSWER", "NEEDS_HELP"], [False, True], [False, True], [False, True])))
def test_placement_truth_table(local, sensitive, online, allow):
    d, _ = placement(local, sensitive, online, allow)
    if local == "ANSWER": assert d == LOCAL
    elif sensitive or not allow: assert d == REFUSE
    elif not online: assert d == QUEUE
    else: assert d == SERVER


# ----------------------------------------------------------------------------------------------- server -> device delivery
def test_server_trains_signs_and_the_device_installs_without_ever_holding_a_key(lab):
    r = teach_iris(lab)
    assert [i[0] for i in r["sync"].installed] == ["iris_species"] and not r["sync"].rejected
    out = lab.client.solve(IRIS_INTENT, load_iris().data[0].tolist())
    assert out["placement"] == LOCAL and out["label"] == "setosa" and out["executed_on"] == "device"
    assert not list((lab.tmp / "dev").rglob("*.private")) and "private" not in (lab.tmp / "dev_trust.json").read_text().lower()
    trust = json.loads(lab.dev_trust.read_text()); assert all(len(base64.b64decode(v)) == 32 for k, v in trust.items() if k != "__revoked__")


def test_remote_and_local_execution_agree(lab):
    teach_iris(lab)
    for i in (0, 60, 120):
        row = load_iris().data[i].tolist()
        assert lab.client.solve(IRIS_INTENT, row)["label"] == lab.app.solve(IRIS_INTENT, row)["label"]


def test_sync_is_idempotent_and_picks_up_new_versions(lab):
    teach_iris(lab)
    again = lab.client.sync(); assert again.installed == [] and ("iris_species", "up to date") in again.skipped
    r2 = lab.client.teach(iris_spec(seed=1), consent_to_upload=True)
    assert r2["version"] == "0.2.0" and r2["sync"].installed == [("iris_species", "0.2.0")]


# ----------------------------------------------------------------------------------------------- privacy
def test_sensitive_or_unconsented_examples_never_leave_the_device(lab):
    before = lab.client.requests_to_server
    assert lab.client.teach(iris_spec(), sensitive=True, consent_to_upload=True)["uploaded"] is False
    assert lab.client.teach(iris_spec(), consent_to_upload=False)["uploaded"] is False
    assert lab.client.requests_to_server == before and lab.app.packages == {}


def test_sensitive_tasks_are_never_sent_to_the_server(lab):
    before = lab.client.requests_to_server
    out = lab.client.solve("translate this private note", [0.1, 0.2], sensitive=True)
    assert out["placement"] == REFUSE and out["result"] == "NEEDS_HELP" and lab.client.requests_to_server == before and lab.client.pending() == []


def test_server_execution_can_be_forbidden_entirely(tmp_path):
    L = Lab(tmp_path, allow_server_solve=False)
    try:
        out = L.client.solve("something unknown", [0.1, 0.2]); assert out["placement"] == REFUSE and L.client.requests_to_server == 0
    finally: L.stop()


# ----------------------------------------------------------------------------------------------- big capability -> server execution
def make_big_cap(lab, tmp):
    import torch, onnxruntime as ort
    torch.manual_seed(0); m = make_mlp(2, 3, (1500, 1500)); data = export_onnx(m, 2)
    sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"]); rng = np.random.default_rng(0); xs = rng.uniform(0, 1, size=(30, 2)).astype(np.float32)
    tests = [(x.tolist(), int(np.argmax(sess.run(None, {"input": x[None]})[0]))) for x in xs]
    p = tmp / "big.cap"
    build_cap(p, capability_id="big_classifier", version="0.1.0", model_bytes=data, params=sum(q.numel() for q in m.parameters()), input_dim=2, labels=["A", "B", "C"], tests=tests,
              keywords=["huge", "classifier"], description="a model larger than the device limit", signer_id="server-build-1", signer_key=lab.app.key, provenance={"source": "test"}, min_accuracy=0.9,
              input_stats={"min": [0, 0], "max": [1, 1], "mean": [0.5, 0.5], "std": [0.29, 0.29]})
    return p, tests


def test_capability_too_big_for_the_device_is_executed_on_the_server(lab, tmp_path):
    p, tests = make_big_cap(lab, tmp_path)
    assert lab.app.register_package(p)["activated"]
    assert lab.app.packages[next(iter(lab.app.packages))]["model_bytes"] > 8 << 20            # the limit applies to the uncompressed model, not the zip
    rep = lab.client.sync()
    assert rep.installed == [] and any("does not fit this device" in why for _, why in rep.skipped)
    direct = lab.local.import_caps(str(p))[0]                                   # and the core itself would refuse it
    assert direct["activated"] is False and direct["failed_step"] == "compatibility"
    x, want = tests[3]
    out = lab.client.solve("huge classifier", x)
    assert out["placement"] == SERVER and out["executed_on"] == "server" and out["result"] == "ANSWER" and out["label_index"] == want
    assert lab.local.list() == []


# ----------------------------------------------------------------------------------------------- offline
def test_offline_local_capabilities_work_unknown_tasks_queue_and_flush_after_reconnect(lab):
    teach_iris(lab)
    port = lab.port; lab.stop()                                                 # network goes away
    row = load_iris().data[75].tolist()
    assert lab.client.solve(IRIS_INTENT, row)["label"] == "versicolor"         # still answers offline
    q = lab.client.solve("something never learned", [0.1, 0.2])
    assert q["placement"] == QUEUE and q["result"] == "NEEDS_HELP" and len(lab.client.pending()) == 1
    t = lab.client.teach(iris_spec(capability_id="iris_again"), consent_to_upload=True)
    assert t["uploaded"] is False and t["queued"] is True and len(lab.client.pending()) == 2
    assert lab.client.flush()["flushed"] == 0 and len(lab.client.pending()) == 2          # still offline: nothing pretended, nothing lost
    lab.start(port=port)                                                       # reconnect
    res = lab.client.flush()
    assert res["flushed"] == 2 and res["still_queued"] == 0 and lab.client.pending() == []
    assert any(c["capability_id"] == "iris_again" for c in lab.local.list())   # the queued teach was uploaded, trained, delivered and installed


# ----------------------------------------------------------------------------------------------- attacks on the delivery path
def patched(client, fn):
    orig = client._http
    client._http = lambda method, path, body=None, raw=False, max_bytes=64 << 20: fn(orig, method, path, body, raw, max_bytes)


def test_tampered_catalog_is_rejected(lab):
    teach_iris(lab)
    def mitm(orig, method, path, body, raw, mb):
        r = orig(method, path, body, raw, mb)
        if path == "/catalog":
            r["body"] = r["body"].replace('"sequence": ', '"sequence": 9')                 # attacker edits the body
        return r
    patched(lab.client, mitm)
    with pytest.raises(DeliveryError, match="signature invalid"): lab.client.sync()


def test_catalog_signed_by_an_unknown_key_is_rejected(lab, tmp_path):
    teach_iris(lab)
    keygen("evil", tmp_path / "evilkeys"); evil = load_private("evil", tmp_path / "evilkeys")
    def mitm(orig, method, path, body, raw, mb):
        r = orig(method, path, body, raw, mb)
        if path == "/catalog": r["signature"] = {"alg": "ed25519", "key_id": "evil", "sig_b64": base64.b64encode(evil.sign(r["body"].encode())).decode()}
        return r
    patched(lab.client, mitm)
    with pytest.raises(DeliveryError, match="untrusted or revoked"): lab.client.sync()


def test_catalog_replay_is_rejected(lab):
    teach_iris(lab)
    old = lab.client._http("GET", "/catalog")                                   # attacker records today's catalog
    lab.client.teach(iris_spec(seed=1), consent_to_upload=True)                # server moves on
    lab.client.sync()
    patched(lab.client, lambda orig, method, path, body, raw, mb: old if path == "/catalog" else orig(method, path, body, raw, mb))
    with pytest.raises(DeliveryError, match="replay"): lab.client.sync()


def test_stale_catalog_is_rejected_as_a_freeze_attack(lab):
    teach_iris(lab)
    lab.client.max_age = 0.0
    time.sleep(0.05)
    with pytest.raises(DeliveryError, match="older than the maximum allowed age"): lab.client.sync()


def test_modified_package_bytes_do_not_match_the_signed_catalog(lab):
    teach_iris(lab)
    lab.local.root.joinpath("registry.json").unlink(); import shutil; shutil.rmtree(lab.local.root / "store")      # fresh device state
    def mitm(orig, method, path, body, raw, mb):
        r = orig(method, path, body, raw, mb)
        if path.startswith("/packages/"): b = bytearray(r); b[len(b) // 2] ^= 0xFF; return bytes(b)
        return r
    patched(lab.client, mitm)
    rep = lab.client.sync()
    assert rep.installed == [] and rep.rejected and "do not match the signed catalog" in rep.rejected[0][1]


def test_package_resigned_by_an_untrusted_key_is_rejected_by_the_core(lab, tmp_path):
    keygen("evil", tmp_path / "evilkeys"); evil = load_private("evil", tmp_path / "evilkeys")
    teach_iris(lab)
    # a server whose catalog is honest but whose package store was swapped for a package signed by an untrusted key (hash pinned in the catalog entry is of the *new* file only if the attacker also holds the catalog key)
    m = lab.app.svc.models["iris_species"]
    bad = tmp_path / "evil.cap"
    build_cap(bad, capability_id="iris_species", version="0.9.0", model_bytes=m["onnx"], params=m["params"], input_dim=4, labels=m["labels"], tests=m["tests"], keywords=["iris"], description="d",
              signer_id="evil", signer_key=evil, provenance={"source": "attacker"}, min_accuracy=0.5, input_stats=m["stats"])
    assert lab.client.local.import_caps(str(bad))[0]["failed_step"] == "signature"


def test_downgrade_offered_by_a_signed_catalog_is_ignored_and_the_core_refuses_it_too(lab):
    teach_iris(lab); lab.client.teach(iris_spec(seed=1), consent_to_upload=True)
    assert lab.local.list()[0]["active_version"] == "0.2.0"
    old = [p for p in lab.app.packages.values() if p["version"] == "0.1.0"][0]
    saved = dict(lab.app.packages); lab.app.packages = {old["package_id"]: old}; lab.app._bump()      # server (mistakenly or maliciously) lists only the old version
    rep = lab.client.sync()
    assert rep.installed == [] and any("possible downgrade" in w for w in rep.warnings) and lab.local.list()[0]["active_version"] == "0.2.0"
    pkg = lab.app.d / "packages" / f"{old['package_id']}.cap"
    assert lab.local.import_caps(str(pkg))[0]["failed_step"] == "version"                              # defence in depth
    lab.app.packages = saved


def test_revocations_are_proposed_by_the_catalog_but_applied_only_by_the_operator(lab, tmp_path):
    keygen("old-key", tmp_path / "oldk"); oldkey = load_private("old-key", tmp_path / "oldk")
    trust = json.loads(lab.dev_trust.read_text()); trust["old-key"] = (tmp_path / "oldk/old-key.public").read_text(); lab.dev_trust.write_text(json.dumps(trust))
    teach_iris(lab)
    m = lab.app.svc.models["iris_species"]; p = tmp_path / "old.cap"
    build_cap(p, capability_id="legacy_cap", version="0.1.0", model_bytes=m["onnx"], params=m["params"], input_dim=4, labels=m["labels"], tests=m["tests"], keywords=["legacy"], description="d",
              signer_id="old-key", signer_key=oldkey, provenance={"source": "t"}, min_accuracy=0.5, input_stats=m["stats"])
    assert lab.local.import_caps(str(p))[0]["activated"]
    lab.app.state["revoked"] = ["old-key"]; lab.app._bump()
    rep = lab.client.sync()
    assert rep.proposed_revocations == ["old-key"]
    assert lab.local.solve("legacy", [5.1, 3.5, 1.4, 0.2])["result"] == "ANSWER"                        # NOT applied automatically (a remote revocation would be a DoS lever)
    assert lab.client.apply_revocations() == ["old-key"]                                                # operator decision
    out = lab.local.solve("legacy", [5.1, 3.5, 1.4, 0.2])
    assert out["result"] == "NEEDS_HELP" and out["reason_code"] == "MODEL_UNAVAILABLE" and "revoked" in out["reason"]
    assert lab.local.solve(IRIS_INTENT, [5.1, 3.5, 1.4, 0.2])["result"] == "ANSWER"                    # capabilities from the still-trusted key keep working


# ----------------------------------------------------------------------------------------------- server hardening
def test_server_endpoints_require_the_token_and_a_local_host_header(lab):
    assert lab.raw("GET", "/health", auth=False)[0] == 200
    for path in ("/catalog", "/packages/x.cap"): assert lab.raw("GET", path, auth=False)[0] == 401
    assert lab.raw("POST", "/solve", {"intent": "x", "input": [1]}, auth=False)[0] == 401
    assert lab.raw("GET", "/catalog", host="evil.example.com")[0] == 403


@pytest.mark.parametrize("name", ["../../etc/passwd", "..%2f..%2fsecrets.cap", "a/b.cap", "x.txt", "", "A" * 300 + ".cap"])
def test_package_downloads_cannot_escape_the_package_directory(lab, name):
    assert lab.raw("GET", f"/packages/{name}")[0] in (404, 400)


@pytest.mark.parametrize("body,code", [(b"{bad", 400), (b"[1]", 400), (json.dumps({"capability_id": "../x", "x": [[1.0]], "y": [0]}).encode(), 400),
                                       (json.dumps({"capability_id": "ok", "x": [[1.0]] * 3, "y": [0]}).encode(), 400), (json.dumps({"capability_id": "ok", "x": [], "y": []}).encode(), 400),
                                       (json.dumps({"capability_id": "ok", "x": [[1.0] * 300] , "y": [0]}).encode(), 400)])
def test_train_endpoint_rejects_malformed_requests_cleanly(lab, body, code):
    status, out = lab.raw("POST", "/train", body)
    assert status == code and b"Traceback" not in out and lab.app.packages == {}


def test_hostile_training_data_is_refused_not_installed(lab):
    rng = np.random.default_rng(0)
    r = lab.client.teach({"capability_id": "noise", "x": rng.uniform(size=(200, 5)).tolist(), "y": rng.integers(0, 2, 200).tolist(), "min_acc": 0.9}, consent_to_upload=True)
    assert r["uploaded"] and r["learned"] is False and "heldout accuracy" in r["reason"] and lab.app.packages == {}
