"""Phase 7 gates: consolidation (duplicates, archive, distillation, rollback), plan optimisation, learned router."""
import dataclasses, json, random, subprocess
import pytest
from conftest import ROOT
from scripts.cli import Cli, AICLI
from scripts import plans as P
from scripts.plan_opt import optimise
from integrations.escalation import Escalator
from integrations.teacher.simulator import TeacherSimulator, splits, sample
from integrations.teacher.provider import HelpRequest
from packages.capbuild import keygen, write_trust, build_cap
from training.service import BuildService
from training import consolidation as C
from training.routing import hash_features, train_router, DIM
from training.datasets.router_data import make_dataset, UNKNOWN
from training.exporters.onnx_export import export_onnx

INTENT = {"compare_numbers": ("compare these two numbers", [0.1, 0.9]), "point_region": ("is this point inside the circular region", [0.1, 0.1]),
          "argmax_position": ("find the position of the largest value", [0.1, 0.2, 0.9, 0.3])}


@pytest.fixture
def world(tmp_path):
    keygen("k", tmp_path / "keys"); write_trust(tmp_path / "trust.json", {"k": (tmp_path / "keys/k.public").read_text()})
    def make(name, learn=3):
        T = TeacherSimulator(); svc = BuildService(tmp_path / "keys", "k", tmp_path / f"b_{name}", T)
        cli = Cli(tmp_path / f"rt_{name}", tmp_path / "trust.json"); esc = Escalator(cli, T, svc, tmp_path / f"e_{name}")
        for t in list(INTENT)[:learn]:
            assert esc.solve(*INTENT[t])["result"] == "ANSWER"
        return T, svc, cli, esc, tmp_path
    return make


def test_feature_hashing_matches_between_python_and_rust(tmp_path):
    (tmp_path / "t.json").write_text("{}")
    for text in ["Compare these two numbers", "is the point inside the circular region?", "", "x y 12 abc_def", "MiXeD   case\tand\npunctuation!!"]:
        out = subprocess.run([str(AICLI), "--root", str(tmp_path / "r"), "--trust", str(tmp_path / "t.json"), "features", "--text", text, "--dim", "256"], capture_output=True, text=True)
        rust, py = json.loads(out.stdout), hash_features(text, 256)
        assert max(abs(a - b) for a, b in zip(rust, py)) < 1e-6


def test_archive_hides_from_routing_and_restore_brings_back(world):
    T, svc, cli, esc, d = world("arch", learn=2)
    cli._run("archive", "point_region")
    assert [c["capability_id"] for c in cli.list()] == ["compare_numbers"]
    assert cli.solve(*INTENT["point_region"])["result"] == "NEEDS_HELP"
    assert any(c["capability_id"] == "point_region" and c["archived"] for c in cli.json("list-all"))
    cli._run("restore", "point_region")
    assert cli.solve(*INTENT["point_region"])["capability_id"] == "point_region"


def test_archive_refused_while_another_capability_depends_on_it(world):
    T, svc, cli, esc, d = world("dep", learn=1)
    m = svc.models["compare_numbers"]
    build_cap(d / "dep.cap", capability_id="wrapper", version="0.1.0", model_bytes=m["onnx"], params=m["params"], input_dim=2, labels=m["labels"], tests=m["tests"],
              keywords=["wrapper"], description="w", signer_id="k", signer_key=svc.key, dependencies=["compare_numbers"], provenance={"source": "t"}, min_accuracy=0.9, input_stats=m["stats"])
    assert cli.import_caps(str(d / "dep.cap"))[0]["activated"]
    p = cli._run("archive", "compare_numbers", check=False)
    assert p.returncode != 0 and "required by" in p.stderr
    cli._run("archive", "wrapper"); cli._run("archive", "compare_numbers")          # allowed once the dependent is gone


def test_forced_capability_probe_does_not_count_as_usage(world):
    T, svc, cli, esc, d = world("nostats", learn=1)
    before = cli.list()[0]["calls"]
    xs = [[0.1, 0.2]] * 20
    C.predictions(cli, "compare_numbers", xs, d)
    assert cli.list()[0]["calls"] == before


def _duplicate(svc, cli, new_id, keywords, seed=999):
    lp = dataclasses.replace(TeacherSimulator(seed=seed).respond(HelpRequest("compare numbers relation", 2)), capability_id=new_id, keywords=keywords)
    used = set(map(tuple, lp.train_x)) | set(map(tuple, lp.validation_x))
    b = svc.build(lp, "0.1.0", {c["capability_id"] for c in cli.list()}, heldout=sample("compare_numbers", 80, 555, exclude=used, unique=True))
    assert b.promoted and cli.import_caps(str(b.cap_path))[0]["activated"]


def test_duplicate_detected_and_merged_with_routing_preserved(world):
    T, svc, cli, esc, d = world("merge", learn=2)
    _duplicate(svc, cli, "compare_values", ["contrast", "magnitude", "difference"])
    rows = C.pairwise_report(cli, d)
    row = next(r for r in rows if {r["a"], r["b"]} == {"compare_numbers", "compare_values"})
    assert row["verdict"] == "duplicate" and row["agreement_in_distribution"] >= 0.98
    assert not any({r["a"], r["b"]} == {"compare_numbers", "point_region"} for r in rows)       # different label spaces are never compared
    suite = [("contrast the magnitude of two values", [0.2, 0.9], "compare_values"), (INTENT["compare_numbers"][0], [0.1, 0.9], "compare_numbers")]
    log = C.merge_duplicate(cli, svc, "compare_numbers", "compare_values", suite, d)
    assert log["merged"] is True
    r = cli.solve("contrast the magnitude of two values", [0.2, 0.9])
    assert (r["capability_id"], r["status"], r["label"]) == ("compare_numbers", "KNOWN", "LESS")
    assert [c["capability_id"] for c in cli.list()] == ["compare_numbers", "point_region"]


def test_merge_refused_when_keeper_fails_the_duplicates_tests(world):
    T, svc, cli, esc, d = world("nomerge", learn=2)
    before = {c["capability_id"]: c["active_version"] for c in cli.list()}
    log = C.merge_duplicate(cli, svc, "compare_numbers", "point_region", [], d)          # different function
    assert log["merged"] is False and {c["capability_id"]: c["active_version"] for c in cli.list()} == before


def test_overlapping_but_different_rules_are_not_called_duplicates(world):
    T, svc, cli, esc, d = world("overlap", learn=2)
    from training.learning_package.spec import TaskSpec, spec_to_learning_package, sample_spec
    small = TaskSpec.from_dict({"capability_id": "point_region__ext", "description": "small", "keywords": ["small", "inner"], "labels": ["INSIDE", "OUTSIDE"], "input_dim": 2,
        "domain": [{"lo": -1.0, "hi": 1.0}] * 2, "label_expr": "0 if x0*x0 + x1*x1 < 0.2 else 1",
        "worked_examples": [{"x": [0.0, 0.0], "y": 0}, {"x": [0.9, 0.9], "y": 1}, {"x": [0.1, 0.2], "y": 0}]})
    lp = spec_to_learning_package(small, "t")
    b = svc.build(lp, "0.1.0", {"point_region", "compare_numbers"}, heldout=sample_spec(small, 400, 7, exclude=set(map(tuple, lp.train_x)) | set(map(tuple, lp.validation_x)), unique=True),
                  verification=sample_spec(small, 300, 4343, unique=True))
    assert cli.import_caps(str(b.cap_path))[0]["activated"]
    row = next(r for r in C.pairwise_report(cli, d) if {r["a"], r["b"]} == {"point_region", "point_region__ext"})
    assert row["verdict"] == "overlapping_different_function" and row["agreement_in_distribution"] < 0.9


def test_compaction_is_gated_and_reversible(world):
    T, svc, cli, esc, d = world("compact", learn=2)
    ex = splits("point_region")["eval"]; cf = d / "ev.jsonl"
    cf.write_text("\n".join(json.dumps({"intent": INTENT["point_region"][0], "input": x, "expected_index": y}) for x, y in zip(*ex)) + "\n")
    before = cli.batch(str(cf))["summary"]["accuracy"]; size_before = next(c for c in cli.list() if c["capability_id"] == "point_region")["model_bytes"]
    stud, rep = C.distill(svc, "point_region", hidden=(16, 16))
    out = C.compact(svc, cli, "point_region", stud, rep)
    assert out["installed"] and out["bytes_after"] < size_before * 0.6 and out["student_heldout"] >= out["teacher_heldout"] - 0.01
    cli.json("rollback", "point_region")
    assert abs(cli.batch(str(cf))["summary"]["accuracy"] - before) < 1e-9
    # a student that is too small must be rejected by the accuracy gate and leave the registry untouched
    tiny, rep2 = C.distill(svc, "compare_numbers", hidden=(2, 2), epochs=40)
    ver = next(c for c in cli.list() if c["capability_id"] == "compare_numbers")["active_version"]
    out2 = C.compact(svc, cli, "compare_numbers", tiny, rep2)
    assert out2["installed"] is False and next(c for c in cli.list() if c["capability_id"] == "compare_numbers")["active_version"] == ver


def test_plan_optimiser_removes_redundancy_without_changing_outputs(rt3, tmp_path):
    plan = P.count_inside("id"); dup = json.loads(json.dumps(plan))
    extra = [dict(n, id=n["id"] + "_dup", out=n["out"] + "_dup") for n in plan["nodes"] if n["op"] == "cap"]
    dup["nodes"] = plan["nodes"][:4] + extra + plan["nodes"][4:]
    opt, st = optimise(dup)
    assert st["applicable"] and st["cap_calls_before"] == 8 and st["cap_calls_after"] == 4
    rng = random.Random(5); cases = [{"p": [rng.uniform(-1, 1) for _ in range(8)]} for _ in range(60)]
    cf = tmp_path / "c.jsonl"; cf.write_text("\n".join(json.dumps({"inputs": c}) for c in cases) + "\n")
    outs = []
    for i, pl in enumerate((dup, opt)):
        pf = tmp_path / f"p{i}.json"; pf.write_text(json.dumps(pl)); outs.append([r["outputs"] for r in rt3.plan_batch(str(pf), str(cf))["results"]])
    assert outs[0] == outs[1]


def test_plan_optimiser_leaves_loops_and_branches_alone():
    for plan in (P.sort4_bubble("id"), P.branch("id")):
        opt, st = optimise(plan)
        assert st["applicable"] is False and opt == plan


def _install_router(cli, svc, caps, d, version, hard=True):
    classes = caps + [UNKNOWN]
    model, rep = train_router(make_dataset(caps, hard_negatives=hard), classes, epochs=120)
    tests = [(hash_features(t), classes.index(c)) for t, c in make_dataset(caps, seed=77, n_per_class=15, hard_negatives=hard)]
    build_cap(d / f"router_{version}.cap", capability_id="__router__", version=version, model_bytes=export_onnx(model, DIM), params=rep["params"], input_dim=DIM, labels=classes,
              tests=tests, keywords=["router"], description="r", signer_id="k", signer_key=svc.key, provenance={"source": "t"}, min_accuracy=0.8, role="router")
    return cli.import_caps(str(d / f"router_{version}.cap"))[0]


def test_learned_router_is_a_signed_artifact_that_routes_but_is_never_routed_to(world):
    T, svc, cli, esc, d = world("lr", learn=3)
    rep = _install_router(cli, svc, list(INTENT), d, "0.1.0")
    assert rep["activated"] and rep["test_accuracy"] >= 0.8
    assert "__router__" not in [c["capability_id"] for c in cli.list()]
    assert any(c["capability_id"] == "__router__" and c["role"] == "router" for c in cli.json("list-all"))
    out = cli._run_detect("full", "solve", "--intent", "which element is the greatest", "--input", "0.1,0.2,0.9,0.3", router="learned")
    assert out["result"] == "ANSWER" and out["capability_id"] == "argmax_position"
    assert cli._run_detect("full", "solve", "--intent", "translate this text into french", "--input", "0.1,0.2", router="learned")["result"] == "NEEDS_HELP"


def test_learned_router_without_installed_router_fails_loudly(world):
    T, svc, cli, esc, d = world("lr0", learn=1)
    p = cli._run("solve", "--intent", "x", "--input", "0.1,0.2", check=False)
    assert p.returncode == 0                                                    # keyword router default
    q = subprocess.run([str(AICLI), "--root", str(cli.root), "--trust", str(cli.trust), "--router", "learned", "solve", "--intent", "x", "--input", "0.1,0.2"], capture_output=True, text=True)
    assert q.returncode != 0 and "no learned router" in q.stderr


def test_learned_router_cannot_route_to_a_capability_added_after_training(world):
    T, svc, cli, esc, d = world("stale", learn=2)
    assert _install_router(cli, svc, ["compare_numbers", "point_region"], d, "0.1.0")["activated"]
    esc.solve(*INTENT["argmax_position"])                                       # new capability learned after the router was trained
    stale = cli._run_detect("full", "solve", "--intent", "find the position of the largest value", "--input", "0.1,0.2,0.9,0.3", router="learned")
    assert stale["result"] == "NEEDS_HELP"
    assert cli.solve("find the position of the largest value", [0.1, 0.2, 0.9, 0.3])["capability_id"] == "argmax_position"   # keyword router adapts immediately


def test_router_training_data_does_not_contain_evaluation_intents():
    from scripts.run_phase7 import router_eval_sets
    from training.datasets.router_data import overlaps
    assert overlaps([r["intent"] for r in router_eval_sets(random.Random(1))], list(INTENT) + ["majority_vote"]) == []
