"""Phase 1 hard gates (security / activation / routing / reload)."""
import json, re
from pathlib import Path
import pytest
from conftest import repack, make_cap, ROOT
from scripts.cli import Cli


def import1(cli, path):
    return cli.import_caps(str(path))[0]


def assert_clean_reject(cli, rep, step):
    assert rep["activated"] is False and rep["failed_step"] == step, rep
    assert cli.list() == []                                  # registry unchanged
    store = cli.root / "store"
    assert not store.exists() or not list(store.glob("*.cap"))  # nothing copied into the store


def test_good_package_activates_through_every_step(env, cli):
    rep = import1(cli, env["good"])
    assert rep["activated"]
    assert [s["step"] for s in rep["steps"]] == ["parse", "signature", "hashes", "compatibility", "sandbox_load", "bundled_tests", "resource_check", "activate"]
    assert rep["test_accuracy"] >= 0.95


def test_corrupted_model_rejected(env, cli, tmp_path):
    data = bytearray(__import__("zipfile").ZipFile(env["good"]).read("model/model.onnx")); data[len(data) // 2] ^= 0xFF
    bad = repack(env["good"], tmp_path / "bad.cap", replace={"model/model.onnx": bytes(data)})
    assert_clean_reject(cli, import1(cli, bad), "hashes")


def test_tampered_manifest_rejected(env, cli, tmp_path):
    import zipfile
    m = zipfile.ZipFile(env["good"]).read("manifest.json").replace(b'"version": "0.1.0"', b'"version": "0.9.0"')
    bad = repack(env["good"], tmp_path / "bad.cap", replace={"manifest.json": m})
    assert_clean_reject(cli, import1(cli, bad), "signature")


def test_untrusted_signer_rejected(env, cli, tmp_path):
    make_cap(env, tmp_path / "rogue.cap", key="rogue")
    assert_clean_reject(cli, import1(cli, tmp_path / "rogue.cap"), "signature")


def test_garbage_file_rejected(cli, tmp_path):
    (tmp_path / "x.cap").write_bytes(b"not a zip at all")
    assert_clean_reject(cli, import1(cli, tmp_path / "x.cap"), "parse")


@pytest.mark.parametrize("over", [
    dict(runtime_min="9.0.0"),
    dict(model_format="tflite"),
    dict(format_version="cap/2"),
    dict(device_caps=["camera"]),
    dict(min_ram_mb=1 << 20),
])
def test_incompatible_rejected(env, cli, tmp_path, over):
    make_cap(env, tmp_path / "inc.cap", **over)
    assert_clean_reject(cli, import1(cli, tmp_path / "inc.cap"), "compatibility")


def test_device_specific_capability_accepted_where_provided_rejected_elsewhere(env, tmp_path):
    make_cap(env, tmp_path / "br.cap", device_caps=["browser"])
    full = Cli(tmp_path / "full", env["trust"], "PC_FULL")
    cons = Cli(tmp_path / "cons", env["trust"], "PC_CONSTRAINED")
    assert full.import_caps(str(tmp_path / "br.cap"))[0]["activated"]
    assert cons.import_caps(str(tmp_path / "br.cap"))[0]["failed_step"] == "compatibility"


def test_failed_bundled_tests_block_activation(env, cli, tmp_path):
    wrong = [(x, (y + 1) % 3) for x, y in env["tests"]]
    make_cap(env, tmp_path / "wrong.cap", tests=wrong, min_accuracy=0.9)
    assert_clean_reject(cli, import1(cli, tmp_path / "wrong.cap"), "bundled_tests")


def test_undeclared_extra_file_rejected(env, cli, tmp_path):
    bad = repack(env["good"], tmp_path / "x.cap", add={"payload/evil.bin": b"x"})
    assert_clean_reject(cli, import1(cli, bad), "hashes")


def test_path_traversal_rejected(env, cli, tmp_path):
    bad = repack(env["good"], tmp_path / "x.cap", add={"../evil": b"x"})
    assert_clean_reject(cli, import1(cli, bad), "parse")


def test_missing_dependency_rejected(env, cli, tmp_path):
    make_cap(env, tmp_path / "dep.cap", dependencies=["not_installed"])
    assert_clean_reject(cli, import1(cli, tmp_path / "dep.cap"), "dependencies")


def test_version_must_increase_and_rollback_works(env, cli, tmp_path):
    assert import1(cli, env["good"])["activated"]
    assert import1(cli, env["good"])["failed_step"] == "version"          # same version again
    make_cap(env, tmp_path / "v2.cap", version="0.2.0")
    assert import1(cli, tmp_path / "v2.cap")["activated"]
    assert cli.list()[0]["active_version"] == "0.2.0"
    assert cli.json("rollback", "compare_numbers")["active_version"] == "0.1.0"
    assert cli.solve("compare numbers", [0.1, 0.9])["version"] == "0.1.0"


def test_unknown_task_needs_help_never_fabricates(env, cli):
    import1(cli, env["good"])
    out = cli.solve("translate this french sentence", [0.1, 0.9])
    assert out["result"] == "NEEDS_HELP" and "label" not in out
    out = cli.solve("compare numbers", [0.1, 0.9, 0.3])                    # right intent, wrong shape
    assert out["result"] == "NEEDS_HELP"


def test_known_task_answers_and_is_correct(env, cli):
    import1(cli, env["good"])
    out = cli.solve("compare numbers", [0.0, 1.0])
    assert out["result"] == "ANSWER" and out["status"] == "KNOWN" and out["label"] == "LESS"


def test_unload_reload_equivalent(env, cli):
    import1(cli, env["good"])
    r = cli.json("reload-check", "compare_numbers", "--intent", "compare numbers", "--input", "0.3,0.3")
    assert r == {"was_loaded": True, "loaded_after_unload": False, "reloaded": True, "identical": True}


def test_store_tampering_detected_on_reload(env, cli):
    import1(cli, env["good"])
    f = next((cli.root / "store").glob("*.cap"))
    f.write_bytes(repack(f, f.with_suffix(".tmp"), replace={"model/model.onnx": b"zzz"}).read_bytes())
    out = cli.solve("compare numbers", [0.1, 0.2])
    assert out["result"] == "NEEDS_HELP" and out["reason_code"] == "MODEL_UNAVAILABLE" and "label" not in out        # refuses to answer from a modified store file


def test_revoked_signer_blocks_new_imports_and_already_installed_capabilities(env, cli, tmp_path):
    assert import1(cli, env["good"])["activated"]
    assert cli.solve("compare numbers", [0.1, 0.9])["result"] == "ANSWER"
    trust = json.loads(Path(env["trust"]).read_text()); trust["__revoked__"] = ["build-svc-1"]
    revoked = tmp_path / "trust_revoked.json"; revoked.write_text(json.dumps(trust))
    after = Cli(cli.root, revoked)                                           # same runtime state, trust root now revokes the signer
    out = after.solve("compare numbers", [0.1, 0.9])
    assert out["result"] == "NEEDS_HELP" and out["reason_code"] == "MODEL_UNAVAILABLE" and "revoked" in out["reason"]
    rep = after.import_caps(str(env["good"]))[0]
    assert rep["activated"] is False and rep["failed_step"] == "signature" and "revoked" in rep["steps"][-1]["detail"]
    ok = Cli(tmp_path / "other", env["trust"])                                # without the revocation the same package still works
    assert ok.import_caps(str(env["good"]))[0]["activated"]


def test_key_rotation_new_signer_works_while_the_old_one_is_revoked(env, tmp_path):
    from packages.capbuild import keygen, load_private
    keygen("build-svc-2", tmp_path / "k2")
    trust = {"build-svc-1": (env["dir"] / "keys/build-svc-1.public").read_text(), "build-svc-2": (tmp_path / "k2/build-svc-2.public").read_text(), "__revoked__": ["build-svc-1"]}
    (tmp_path / "t.json").write_text(json.dumps(trust))
    make_cap(env, tmp_path / "new.cap", key="rogue") if False else None
    from packages.capbuild import build_cap
    build_cap(tmp_path / "rot.cap", capability_id="compare_numbers", version="0.3.0", model_bytes=env["onnx"], params=1251, input_dim=2, labels=["LESS", "EQUAL", "GREATER"], tests=env["tests"],
              keywords=env["lp"].keywords, description="d", signer_id="build-svc-2", signer_key=load_private("build-svc-2", tmp_path / "k2"), provenance={"source": "t"}, min_accuracy=0.9)
    c = Cli(tmp_path / "rt", tmp_path / "t.json")
    assert c.import_caps(str(tmp_path / "rot.cap"))[0]["activated"]            # signed by the new key
    assert c.import_caps(str(env["good"]))[0]["failed_step"] == "signature"    # old key revoked


def test_teacher_side_has_no_signing_access():
    import ast
    banned_mods = ("packages", "cryptography", "training.service")
    banned_names = {"load_private", "build_cap", "keygen", "Ed25519PrivateKey", "BuildService"}
    for f in (ROOT / "integrations").rglob("*.py"):
        tree = ast.parse(f.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                assert not any(a.name.startswith(banned_mods) for a in n.names), f
            elif isinstance(n, ast.ImportFrom):
                assert not (n.module or "").startswith(banned_mods), f
            elif isinstance(n, ast.Name):
                assert n.id not in banned_names, f


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_inputs_are_refused_never_answered(env, cli, bad):
    import1(cli, env["good"])
    out = cli.solve("compare numbers", [0.2, bad])
    assert out["result"] == "NEEDS_HELP" and out["reason_code"] == "INVALID_INPUT" and "label" not in out
