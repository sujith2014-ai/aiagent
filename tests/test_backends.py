"""Backends: the pure-Rust mlp-lite executor must agree with the general runtime (tract) through the whole system, report which backend ran,
reject what it does not understand, and the general runtime must remain the fallback."""
import json, random
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper
import pytest
from conftest import ROOT
from scripts.cli import Cli
from packages.capbuild import build_cap
from training.learning_package.examples import package_from_examples
from sklearn.datasets import load_iris, load_wine

INTENTS = ["compare these two numbers", "is this point inside the circular region", "find the position of the largest value"]


def batch(cli, cases, backend):
    return cli._run_detect("full", "batch", "--cases", str(cases), backend=backend)["results"]


def agree(a, b):
    assert len(a) == len(b)
    worst = 0.0
    for x, y in zip(a, b):
        assert bool(x.get("needs_help")) == bool(y.get("needs_help"))
        if x.get("needs_help"): continue
        assert x["label_index"] == y["label_index"] and x["status"] == y["status"]
        worst = max(worst, max(abs(p - q) for p, q in zip(x["probs"], y["probs"])))
    return worst


def test_backends_agree_on_the_three_synthetic_capabilities_through_the_runtime(rt3, tmp_path):
    from integrations.teacher.simulator import splits
    for cap, intent in zip(("compare_numbers", "point_region", "argmax_position"), INTENTS):
        xs, ys = splits(cap)["eval"]
        f = tmp_path / f"{cap}.jsonl"; f.write_text("\n".join(json.dumps({"intent": intent, "input": x, "expected_index": y}) for x, y in zip(xs[:200], ys[:200])) + "\n")
        lite, tract = batch(rt3, f, "mlp"), batch(rt3, f, "tract")
        assert agree(lite, tract) < 1e-5, cap


def test_backends_agree_on_real_data_models_with_baked_in_standardisation(env, tmp_path):
    for name, loader in (("iris", load_iris), ("wine", load_wine)):
        d = loader(); lp, (xte, yte), _ = package_from_examples(name, f"classify {name}", [name], [str(t) for t in d.target_names], d.data.tolist(), d.target.tolist())
        b = env["svc"].build(lp, "0.1.0", set(), heldout=(xte, yte), standardize=True, min_acc=0.9, attempts=((32, 32), (64, 64)))
        assert b.promoted
        cli = Cli(tmp_path / f"rt_{name}", env["trust"]); assert cli.import_caps(str(b.cap_path))[0]["activated"]
        f = tmp_path / f"{name}.jsonl"; f.write_text("\n".join(json.dumps({"intent": f"classify {name}", "input": x, "expected_index": y}) for x, y in zip(xte, yte)) + "\n")
        assert agree(batch(cli, f, "mlp"), batch(cli, f, "tract")) < 1e-5, name


def test_import_reports_which_backend_ran_the_model(env, tmp_path):
    for backend, want in (("mlp", "onnx-mlp-lite"), ("tract", "onnx-tract"), ("auto", "onnx-mlp-lite")):
        cli = Cli(tmp_path / f"rt_{backend}", env["trust"])
        out = cli._run_detect("full", "import", str(env["good"]), backend=backend)
        step = next(s for s in out[0]["steps"] if s["step"] == "sandbox_load")
        assert out[0]["activated"] and step["detail"] == want


def tanh_cap(env, tmp_path):
    """A valid, signed package whose model uses an operator (Tanh) that mlp-lite does not implement."""
    rng = np.random.default_rng(0)
    w1, b1, w2, b2 = rng.normal(size=(8, 2)).astype(np.float32), rng.normal(size=8).astype(np.float32), rng.normal(size=(3, 8)).astype(np.float32), rng.normal(size=3).astype(np.float32)
    g = helper.make_graph(
        [helper.make_node("Gemm", ["input", "w1", "b1"], ["h"], transB=1), helper.make_node("Tanh", ["h"], ["t"]), helper.make_node("Gemm", ["t", "w2", "b2"], ["logits"], transB=1)], "g",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])], [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])],
        [numpy_helper.from_array(a, n) for a, n in ((w1, "w1"), (b1, "b1"), (w2, "w2"), (b2, "b2"))])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)]); m.ir_version = 7
    data = m.SerializeToString(); sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    xs = rng.uniform(0, 1, size=(60, 2)).astype(np.float32)
    tests = [(x.tolist(), int(np.argmax(sess.run(None, {"input": x[None]})[0]))) for x in xs]
    path = tmp_path / "tanh.cap"
    build_cap(path, capability_id="tanh_net", version="0.1.0", model_bytes=data, params=int(w1.size + b1.size + w2.size + b2.size), input_dim=2, labels=["A", "B", "C"], tests=tests,
              keywords=["tanh"], description="t", signer_id="build-svc-1", signer_key=env["keyfn"]("build-svc-1"), provenance={"source": "test"}, min_accuracy=0.9,
              input_stats={"min": [0, 0], "max": [1, 1], "mean": [0.5, 0.5], "std": [0.29, 0.29]})
    return path, tests


def test_unsupported_operator_is_rejected_by_mlp_lite_and_handled_by_the_fallback(env, tmp_path):
    path, tests = tanh_cap(env, tmp_path)
    lite = Cli(tmp_path / "rt_lite", env["trust"])
    rep = lite._run_detect("full", "import", str(path), backend="mlp")[0]
    assert rep["activated"] is False and rep["failed_step"] == "sandbox_load" and "unsupported operator 'Tanh'" in rep["steps"][-1]["detail"]
    assert lite.list() == []                                                          # nothing installed
    auto = Cli(tmp_path / "rt_auto", env["trust"])
    rep = auto._run_detect("full", "import", str(path), backend="auto")[0]
    assert rep["activated"] is True and next(s for s in rep["steps"] if s["step"] == "sandbox_load")["detail"] == "onnx-tract"   # fell back to the general runtime
    out = auto._run_detect("full", "solve", "--intent", "tanh", "--input", "0.3,0.6", backend="auto")
    assert out["result"] == "ANSWER"


def test_a_signed_but_malformed_model_is_rejected_cleanly(env, tmp_path):
    """Hashes and signature protect integrity, not sanity: a validly signed garbage model must still fail at sandbox_load, never crash."""
    from packages.capbuild import build_cap as bc
    for label, data in (("garbage", b"definitely not onnx"), ("empty_graph", b"\x3a\x00"), ("truncated", Path(ROOT / "core/tests/fixtures/mlp_compare.onnx").read_bytes()[:200])):
        p = tmp_path / f"{label}.cap"
        bc(p, capability_id="x", version="0.1.0", model_bytes=data, params=1, input_dim=2, labels=["A", "B"], tests=[([0.1, 0.2], 0)], keywords=["x"], description="d",
           signer_id="build-svc-1", signer_key=env["keyfn"]("build-svc-1"), provenance={"source": "t"}, min_accuracy=0.0)
        for backend in ("mlp", "tract", "auto"):
            cli = Cli(tmp_path / f"rt_{label}_{backend}", env["trust"])
            rep = cli._run_detect("full", "import", str(p), backend=backend)[0]
            assert rep["activated"] is False and rep["failed_step"] == "sandbox_load", (label, backend, rep)
            assert cli.list() == []
