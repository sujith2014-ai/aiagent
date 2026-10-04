"""Phase 6 gates: safe rule evaluation, real-provider HTTP path (mock), environment verification, selective learning."""
import json, hashlib, os, zipfile
import pytest
from conftest import ROOT
from scripts.cli import Cli
from integrations.escalation import Escalator
from integrations.teacher import mock_server
from integrations.teacher.http_providers import OpenAICompatProvider, AnthropicProvider
from integrations.teacher.provider import TeacherProvider
from integrations.teacher.simulator import TeacherSimulator, splits
from packages.capbuild import keygen, write_trust
from training.safe_expr import compile_expr, evaluate, names_for, UnsafeExpression
from training.service import BuildService
from training.strategies import SelectiveLearner
from training.learning_package.spec import TaskSpec, sample_spec

INTENTS = {"compare_numbers": ("compare these two numbers", [0.1, 0.9]), "point_region": ("is this point inside the circular region", [0.1, 0.1]),
           "argmax_position": ("find the position of the largest value", [0.1, 0.2, 0.9, 0.3])}


@pytest.fixture
def world(tmp_path):
    keygen("k", tmp_path / "keys"); write_trust(tmp_path / "trust.json", {"k": (tmp_path / "keys/k.public").read_text()})
    def make(name, teacher=None, learn=0):
        T = teacher or TeacherSimulator()
        svc = BuildService(tmp_path / "keys", "k", tmp_path / f"b_{name}", T)
        cli = Cli(tmp_path / f"rt_{name}", tmp_path / "trust.json")
        esc = Escalator(cli, T, svc, tmp_path / f"e_{name}")
        for t in list(INTENTS)[:learn]:
            assert esc.solve(*INTENTS[t])["result"] == "ANSWER"
        return T, svc, cli, esc
    return make


def env(task, n=200):
    ex = splits(task)["eval"]; return ex[0][:n], ex[1][:n]


@pytest.mark.parametrize("expr", ["__import__('os').system('id')", "().__class__", "x0.real", "[i for i in x]", "lambda: 1", "open('f')", "2**10000",
                                  "eval('1')", "'abc'", "x[0]**99", "print(1)", "a+1", "x0 if x0 else __builtins__", "(lambda: 1)()", "x0 @ x1"])
def test_hostile_expressions_rejected(expr):
    with pytest.raises(UnsafeExpression):
        evaluate(compile_expr(expr, names_for(2)), [1.0, 2.0])


def test_safe_expression_semantics():
    t = compile_expr("0 if x0 < x1 else 1 if x0 == x1 else 2", names_for(2))
    assert [evaluate(t, v) for v in ([1, 2], [2, 2], [3, 2])] == [0, 1, 2]
    assert evaluate(compile_expr("max(x0, x1, x2) - min(x)", names_for(3)), [1, 5, 2]) == 4


def test_openai_compatible_provider_end_to_end_over_http(world):
    backend = TeacherSimulator(mode="spec")
    srv, url, seen = mock_server.start(backend); os.environ["MOCK_KEY"] = "sk-test-secret-value"
    try:
        T, svc, cli, esc = world("http", teacher=OpenAICompatProvider(url, "m", api_key_env="MOCK_KEY"))
        svc.oracle = backend
        r = esc.solve(*INTENTS["compare_numbers"], examples=env("compare_numbers"))
    finally:
        srv.shutdown()
    assert r["result"] == "ANSWER" and r["path"] == ["local", "teacher", "learn"]
    assert seen[0]["headers"]["authorization"] == "Bearer sk-test-secret-value"
    body = seen[0]["body"]
    assert "sk-test-secret-value" not in json.dumps(body)
    assert set(json.loads(body["messages"][-1]["content"])) == {"task_intent", "input_dim", "known_capabilities", "attempted_route", "failure", "available_tools"}


def test_anthropic_style_provider_end_to_end_over_http(world):
    backend = TeacherSimulator(mode="spec")
    srv, url, seen = mock_server.start(backend); os.environ["MOCK_KEY"] = "k"
    try:
        T, svc, cli, esc = world("anth", teacher=AnthropicProvider("m", base_url=url, api_key_env="MOCK_KEY"))
        r = esc.solve(*INTENTS["point_region"], examples=env("point_region"))
    finally:
        srv.shutdown()
    assert r["result"] == "ANSWER" and seen[0]["path"] == "/v1/messages" and "x-api-key" in seen[0]["headers"]


def test_http_failure_queues_and_does_not_count_a_teacher_call(world):
    srv, url, _ = mock_server.start(TeacherSimulator(), fail=True)
    try:
        T, svc, cli, esc = world("fail", teacher=OpenAICompatProvider(url, "m"))
        r = esc.solve(*INTENTS["compare_numbers"])
    finally:
        srv.shutdown()
    assert r["result"] == "QUEUED_OFFLINE" and esc.teacher_calls == 0 and cli.list() == []


def test_garbage_text_from_provider_is_rejected(world):
    srv, url, _ = mock_server.start(TeacherSimulator(), garbage="Sure! DROP TABLE users;")
    try:
        T, svc, cli, esc = world("garb", teacher=OpenAICompatProvider(url, "m"))
        r = esc.solve(*INTENTS["compare_numbers"])
    finally:
        srv.shutdown()
    assert r["result"] == "NEEDS_HELP" and cli.list() == []


def test_spec_without_environment_examples_cannot_be_verified(world):
    T, svc, cli, esc = world("noex", teacher=TeacherSimulator(mode="spec"))
    r = esc.solve(*INTENTS["compare_numbers"])
    assert r["result"] == "NEEDS_HELP" and "verified" in r["reason"] and cli.list() == []


@pytest.mark.parametrize("fault", ["contradicts_examples", "malicious_expr", "plausible_wrong"])
def test_faulty_teacher_rules_are_not_installed(world, fault):
    T, svc, cli, esc = world(f"f_{fault}", teacher=TeacherSimulator(mode="spec", fault=fault))
    r = esc.solve(*INTENTS["point_region"], examples=env("point_region"))
    assert r["result"] == "NEEDS_HELP" and cli.list() == []


def test_subtle_teacher_error_below_tolerance_is_a_known_limitation(world):
    """Documents a limit, not a feature: a rule wrong on ~1% of inputs passes a 0.95 gate."""
    T, svc, cli, esc = world("subtle", teacher=TeacherSimulator(mode="spec", fault="subtle_wrong"))
    assert esc.solve(*INTENTS["compare_numbers"], examples=env("compare_numbers"))["result"] == "ANSWER"


def _model_hash(cap_path):
    return hashlib.sha256(zipfile.ZipFile(cap_path).read("model/model.onnx")).hexdigest()


def test_router_update_persists_alias_without_neural_change_and_removes_teacher_dependency(world):
    T, svc, cli, esc = world("ru", learn=1)
    before = {c["capability_id"]: c for c in cli.list()}["compare_numbers"]
    calls = esc.teacher_calls
    r = esc.solve("order the two values", [0.9, 0.1], examples=env("compare_numbers"))
    assert r["result"] == "ANSWER" and "router_update" in r["path"] and esc.teacher_calls == calls + 1
    after = {c["capability_id"]: c for c in cli.list()}["compare_numbers"]
    assert after["active_version"] != before["active_version"] and "order" in after["keywords"]
    caps = sorted((cli.root / "store").glob("compare_numbers*.cap"))
    assert len(caps) == 2 and _model_hash(caps[0]) == _model_hash(caps[1])           # same model bytes: no neural change
    r2 = esc.solve("order the two values", [0.2, 0.8])
    assert r2["path"] == ["local"] and esc.teacher_calls == calls + 1                 # second time: no teacher


class Scripted(TeacherProvider):
    name = "scripted"
    def __init__(self, d): self.d = d
    def advise(self, req): return json.dumps(self.d)


def test_router_update_rejected_when_capability_fails_environment_examples(world):
    T, svc, cli, esc = world("poison", learn=2)
    esc.teacher = Scripted({"action": "reroute", "reroute_intent": "compare numbers relation", "capability_id": "compare_numbers", "rationale": "x"})
    before = {c["capability_id"]: c["active_version"] for c in cli.list()}
    r = esc.solve("decide which side this location is on", [0.1, 0.1], examples=env("point_region"))   # no keyword overlap -> escalates; examples are *region* examples
    assert r["router_update"]["accepted"] is False and "environment examples" in r["router_update"]["why"]
    assert {c["capability_id"]: c["active_version"] for c in cli.list()} == before


def test_router_update_that_hijacks_other_intents_is_rolled_back(world):
    T, svc, cli, esc = world("hijack", learn=2)
    esc.solve(*INTENTS["point_region"])    # ensure routing suite holds the region intent
    esc.teacher = Scripted({"action": "reroute", "reroute_intent": "compare numbers relation", "capability_id": "compare_numbers", "rationale": "x"})
    before = {c["capability_id"]: c["active_version"] for c in cli.list()}
    long_intent = ("point inside circular region " + " ".join(f"filler{i}x" for i in range(18)))   # contains region words but dilutes coverage below the routing threshold
    r = esc.solve(long_intent, [0.2, 0.8], examples=env("compare_numbers"))
    assert r["router_update"]["accepted"] is False and "routing regression" in r["router_update"]["why"]
    assert {c["capability_id"]: c["active_version"] for c in cli.list()} == before
    assert cli.solve(*INTENTS["point_region"])["capability_id"] == "point_region"


WIDE = {"capability_id": "compare_numbers", "description": "Compare two numbers (wide range)", "keywords": ["compare", "numbers"], "labels": ["LESS", "EQUAL", "GREATER"],
        "input_dim": 2, "domain": [{"lo": 0.0, "hi": 79 / 19, "grid": 80}] * 2, "label_expr": "0 if x0 < x1 else 1 if x0 == x1 else 2",
        "worked_examples": [{"x": [0.5, 3.0], "y": 0}, {"x": [3.0, 3.0], "y": 1}, {"x": [4.0, 1.0], "y": 2}]}


def test_domain_extension_prefers_adaptation_and_preserves_old_domain(world):
    T, svc, cli, esc = world("ext", learn=1)
    assert cli.solve("compare these two numbers", [3.1, 2.6])["reason_code"] == "OUT_OF_DISTRIBUTION"
    spec = TaskSpec.from_dict(WIDE); venv = sample_spec(spec, 300, 4242, unique=True); old = splits("compare_numbers")["eval"]
    log = SelectiveLearner(svc, cli).extend_domain("compare_numbers", spec, venv, old)
    assert log["chosen"] == "adapt_existing"
    assert cli.solve("compare these two numbers", [3.1, 2.6])["label"] == "GREATER"
    assert cli.solve("compare these two numbers", [0.1, 0.9])["label"] == "LESS"
    assert cli.regression("compare_numbers")["bundled_test_accuracy"] >= 0.95


def test_conflicting_rule_falls_back_to_new_module_and_old_capability_survives(world):
    T, svc, cli, esc = world("conf", learn=2)
    small = {"capability_id": "point_region", "description": "small inner circle", "keywords": ["small", "inner", "core", "circle", "point", "inside", "region"],
             "labels": ["INSIDE", "OUTSIDE"], "input_dim": 2, "domain": [{"lo": -1.0, "hi": 1.0}] * 2, "label_expr": "0 if x0*x0 + x1*x1 < 0.2 else 1",
             "worked_examples": [{"x": [0.0, 0.0], "y": 0}, {"x": [0.9, 0.9], "y": 1}, {"x": [0.1, 0.2], "y": 0}]}
    spec = TaskSpec.from_dict(small); venv = sample_spec(spec, 300, 4343, unique=True)
    log = SelectiveLearner(svc, cli).extend_domain("point_region", spec, venv, splits("point_region")["eval"], new_cap_keywords=small["keywords"])
    assert log["chosen"] == "new_module" and log["new_capability_id"] == "point_region__ext"
    adapt = next(a for a in log["attempts"] if a["strategy"] == "adapt_existing")
    assert adapt["passed"] is False                                              # adaptation could not satisfy both rules
    old = cli.solve("is this point inside the circular region", [0.5, 0.0]); new = cli.solve("is this point inside the small inner region", [0.5, 0.0])
    assert (old["capability_id"], old["label"]) == ("point_region", "INSIDE")
    assert (new["capability_id"], new["label"]) == ("point_region__ext", "OUTSIDE")
