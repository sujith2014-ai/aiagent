"""Phase 4 gates: dynamic composition, reuse, branching, loops with safety limits, NEEDS_HELP inside plans."""
import json, random
from scripts import plans as P

G = 19


def run(cli, plan, cases, tmp_path):
    pf, cf = tmp_path / "plan.json", tmp_path / "cases.jsonl"
    pf.write_text(json.dumps(plan)); cf.write_text("\n".join(json.dumps({"inputs": c}) for c in cases) + "\n")
    return cli.plan_batch(str(pf), str(cf))


def test_sort_network_sorts_and_reuses_one_module_five_times(rt3, tmp_path):
    rng = random.Random(1)
    cases = [{"v": [k / G for k in rng.sample(range(20), 4)]} for _ in range(100)]
    out = run(rt3, P.sort4_network("intent"), cases, tmp_path)
    assert all(r["outputs"] == sorted(c["v"]) or all(abs(a - b) < 1e-5 for a, b in zip(r["outputs"], sorted(c["v"]))) for r, c in zip(out["results"], cases))
    assert all(r["calls"] == {"compare_numbers": 5} for r in out["results"])


def test_loop_version_reuses_module_nine_times(rt3, tmp_path):
    cases = [{"v": [0.9, 0.3, 0.6, 0.1]}]
    r = run(rt3, P.sort4_bubble("intent"), cases, tmp_path)["results"][0]
    assert r["calls"] == {"compare_numbers": 9}
    assert [round(x, 4) for x in r["outputs"]] == [0.1, 0.3, 0.6, 0.9]


def test_branch_executes_only_the_taken_side(rt3, tmp_path):
    cases = [{"a": [0.9, 0.1], "p": [0.0, 0.0], "v": [0.1, 0.2, 0.3, 0.4]},   # a>b -> region branch
             {"a": [0.1, 0.9], "p": [0.0, 0.0], "v": [0.1, 0.2, 0.3, 0.4]}]   # else -> argmax branch
    out = run(rt3, P.branch("intent"), cases, tmp_path)["results"]
    nodes = [[x["node"] for x in r["records"]] for r in out]
    assert "tb" in nodes[0] and "ec" not in nodes[0]
    assert "ec" in nodes[1] and "tb" not in nodes[1]


def test_unknown_intent_inside_plan_returns_needs_help_not_a_guess(rt3, tmp_path):
    plan = {"id": "x", "inputs": {"v": 2}, "nodes": [P.cap("n", "translate french sentence", [P.S("v")], "o")], "outputs": [P.S("o")]}
    r = run(rt3, plan, [{"v": [0.1, 0.2]}], tmp_path)["results"][0]
    assert r.get("needs_help") and "outputs" not in r


def test_ambiguous_intent_is_not_silently_resolved(rt3, tmp_path):
    plan = {"id": "x", "inputs": {"v": 2}, "nodes": [P.cap("n", "compare region", [P.S("v")], "o")], "outputs": [P.S("o")]}
    assert run(rt3, plan, [{"v": [0.1, 0.2]}], tmp_path)["results"][0].get("needs_help")


def test_missing_capability_id_is_a_clean_error(rt3, tmp_path):
    plan = {"id": "x", "inputs": {"v": 2}, "nodes": [P.cap("n", "nope", [P.S("v")], "o", by="id")], "outputs": [P.S("o")]}
    r = run(rt3, plan, [{"v": [0.1, 0.2]}], tmp_path)["results"][0]
    assert "error" in r and "unknown capability" in r["error"]


def test_invalid_plans_rejected_statically(rt3, tmp_path):
    for plan in [
        {"id": "u", "inputs": {"v": 1}, "nodes": [{"op": "gather", "id": "a", "args": [P.S("zz")], "out": "o"}], "outputs": [P.S("o")]},       # undefined slot
        {"id": "c", "inputs": {"v": 1}, "nodes": [{"op": "gather", "id": "a", "args": [P.S("o")], "out": "o"}], "outputs": [P.S("o")]},        # self-dependency (cycle)
        {"id": "r", "inputs": {"v": 1}, "nodes": [{"op": "repeat", "id": "r", "times": 10**6, "body": []}], "outputs": [P.S("v")]},           # unbounded loop
        {"id": "d", "inputs": {"v": 1}, "nodes": [{"op": "gather", "id": "a", "args": [P.S("v")], "out": "o"}, {"op": "gather", "id": "a", "args": [P.S("v")], "out": "p"}], "outputs": [P.S("o")]},
    ]:
        pf = tmp_path / "p.json"; pf.write_text(json.dumps(plan))
        assert rt3.plan_validate(str(pf))["valid"] is False


def test_nested_loops_hit_the_runtime_step_budget(rt3, tmp_path):
    plan = {"id": "n", "inputs": {"v": 1}, "nodes": [{"op": "repeat", "id": "o", "times": 1000, "body": [{"op": "repeat", "id": "i", "times": 1000,
            "body": [{"op": "gather", "id": "g", "args": [P.S("v")], "out": "v"}]}]}], "outputs": [P.S("v")]}
    r = run(rt3, plan, [{"v": [1.0]}], tmp_path)["results"][0]
    assert "budget" in r["error"]


def test_wrong_input_shape_rejected(rt3, tmp_path):
    r = run(rt3, P.count_inside("intent"), [{"p": [0.1, 0.2]}], tmp_path)["results"][0]
    assert "error" in r
