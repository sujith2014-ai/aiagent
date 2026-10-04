"""Phase 7: consolidation/pruning experiments and learned-vs-deterministic routing."""
from __future__ import annotations
import dataclasses, json, random, shutil, sys, time, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from scripts.cli import Cli, ROOT
from scripts import plans as P
from scripts.plan_opt import optimise
from packages.capbuild import keygen, write_trust, build_cap
from integrations.teacher.simulator import TeacherSimulator, splits, TASKS, sample
from integrations.teacher.provider import HelpRequest
from integrations.escalation import Escalator
from training.service import BuildService
from training import consolidation as C
from training.routing import train_router, hash_features, DIM
from training.datasets.router_data import make_dataset, overlaps, UNKNOWN
from training.exporters.onnx_export import export_onnx
from training.learning_package.spec import TaskSpec, spec_to_learning_package, sample_spec
from scripts.run_phase5 import CANON, PARA, OOS, ADV

RUN = ROOT / "runs" / "phase7"
INTENT = {"compare_numbers": "compare these two numbers", "point_region": "is this point inside the circular region",
          "argmax_position": "find the position of the largest value", "majority_vote": "decide whether most of the values exceed one half"}


def inputs_for(task, rng, n=1):
    if task == "point_region":
        return [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(n)]
    d = TASKS[task]["dim"]
    if task == "compare_numbers":
        return [[rng.randrange(20) / 19, rng.randrange(20) / 19] for _ in range(n)]
    return [[rng.random() for _ in range(d)] for _ in range(n)]


def learn(task, esc):
    r = esc.solve(INTENT[task], inputs_for(task, random.Random(1))[0]); assert r["result"] == "ANSWER", r
    return r


def total_stats(cli):
    caps = cli.json("list-all")
    act = [c for c in caps if c["role"] == "capability" and not c["archived"]]
    return {"active_capabilities": len(act), "archived": sum(c["archived"] for c in caps), "active_params": sum(c["params"] for c in act),
            "active_model_bytes": sum(c["model_bytes"] for c in act)}


# ---------------------------------------------------------------- A. consolidation
def consolidation(trust):
    out = {}
    teacher = TeacherSimulator(); svc = BuildService(RUN / "keys", "build-svc-1", RUN / "build_cons", teacher)
    cli = Cli(RUN / "rt_cons", trust); esc = Escalator(cli, teacher, svc, RUN / "esc_cons")
    for t in ("compare_numbers", "point_region", "argmax_position", "majority_vote"):
        learn(t, esc)
    # a duplicate learned separately under another id/keywords (different teacher sample => different weights)
    t2 = TeacherSimulator(seed=999)
    lp = dataclasses.replace(t2.respond(HelpRequest("compare numbers relation", 2)), capability_id="compare_values",
                             keywords=["contrast", "values", "magnitude", "pair", "difference"], description="Contrast the magnitude of two values")
    used = set(map(tuple, lp.train_x)) | set(map(tuple, lp.validation_x))
    b = svc.build(lp, "0.1.0", set(c["capability_id"] for c in cli.list()), heldout=sample("compare_numbers", 80, 555, exclude=used, unique=True))
    assert b.promoted, b.reason
    assert cli.import_caps(str(b.cap_path))[0]["activated"]
    # an overlapping-but-different capability (rule changed: small circle) as its own module
    small = TaskSpec.from_dict({"capability_id": "point_region__ext", "description": "small inner circle", "keywords": ["small", "inner", "core", "circle", "point", "inside", "region"],
        "labels": ["INSIDE", "OUTSIDE"], "input_dim": 2, "domain": [{"lo": -1.0, "hi": 1.0}] * 2, "label_expr": "0 if x0*x0 + x1*x1 < 0.2 else 1",
        "worked_examples": [{"x": [0.0, 0.0], "y": 0}, {"x": [0.9, 0.9], "y": 1}, {"x": [0.1, 0.2], "y": 0}]})
    lps = spec_to_learning_package(small, "t"); venv = sample_spec(small, 300, 4343, unique=True)
    b = svc.build(lps, "0.1.0", set(c["capability_id"] for c in cli.list()), heldout=sample_spec(small, 400, 7, exclude=set(map(tuple, lps.train_x)) | set(map(tuple, lps.validation_x)), unique=True), verification=venv)
    assert cli.import_caps(str(b.cap_path))[0]["activated"]
    # a capability that declares a dependency on compare_numbers (guards archiving)
    m = svc.models["compare_numbers"]
    dep = RUN / "dep.cap"
    build_cap(dep, capability_id="compare_wrapper", version="0.1.0", model_bytes=m["onnx"], params=m["params"], input_dim=2, labels=m["labels"], tests=m["tests"],
              keywords=["wrapper", "pairwise"], description="wrapper", signer_id="build-svc-1", signer_key=svc.key, dependencies=["compare_numbers"],
              provenance={"source": "experiment"}, min_accuracy=0.9, input_stats=m["stats"])
    assert cli.import_caps(str(dep))[0]["activated"]
    out["before"] = total_stats(cli)
    # --- traffic (populates usage statistics; the routing suite records served intents)
    rng = random.Random(3); suite = []
    for t, n in (("compare_numbers", 120), ("point_region", 100), ("argmax_position", 80)):
        for x in inputs_for(t, rng, n):
            r = cli.solve(INTENT[t], x); assert r["result"] == "ANSWER"
            if len(suite) < 60 and r["status"] == "KNOWN":
                suite.append((INTENT[t], x, r["capability_id"]))
    for x in inputs_for("compare_numbers", rng, 20):
        r = cli.solve("contrast the magnitude of two values", x); suite.append(("contrast the magnitude of two values", x, r["capability_id"]))
    for x in inputs_for("point_region", rng, 5):
        cli.solve("is this point inside the small inner region", x)
    out["routing_suite_size"] = len(suite)
    # --- detection
    out["pairwise_functional_agreement"] = C.pairwise_report(cli, RUN)
    # --- merge duplicate
    ml = C.merge_duplicate(cli, svc, "compare_numbers", "compare_values", suite, RUN)
    r_after = cli.solve("contrast the magnitude of two values", [0.2, 0.9])
    ml["dup_intent_now_routes_to"] = {"capability": r_after.get("capability_id"), "status": r_after.get("status"), "label": r_after.get("label")}
    cli._run("restore", "compare_values"); restored = cli.solve("contrast the magnitude of two values", [0.2, 0.9]); cli._run("archive", "compare_values")
    ml["restore_works"] = restored.get("capability_id") in ("compare_values", "compare_numbers")
    out["merge_duplicate"] = ml
    # --- archive guard + archive unused
    p = cli._run("archive", "compare_numbers", check=False)
    out["archive_guard"] = {"refused": p.returncode != 0, "message": p.stderr.strip()[:160]}
    out["archive_unused"] = C.archive_unused(cli, min_calls=3)
    # --- distillation / pruning
    comp = []
    for cap in ("compare_numbers", "point_region", "argmax_position"):
        ex = splits(cap)["eval"]; cf = RUN / f"eval_{cap}.jsonl"
        cf.write_text("\n".join(json.dumps({"intent": INTENT[cap], "input": x, "expected_index": y, "expected_capability": cap}) for x, y in zip(*ex)) + "\n")
        before = cli.batch(str(cf))["summary"]; entry = {"capability": cap, "before": {"accuracy": before["accuracy"], "latency_us_p50": before["latency_us_p50"],
                 "params": svc.models[cap]["params"], "bytes": next(c for c in cli.list() if c["capability_id"] == cap)["model_bytes"]}}
        stud, rep = C.distill(svc, cap, hidden=(16, 16)); prn, prep = C.prune_structured(svc, cap, keep=0.5)
        entry["pruning_dry_run"] = {k: prep[k] for k in ("teacher_heldout", "student_heldout", "params_before", "params_after")}
        inst = C.compact(svc, cli, cap, stud, rep)
        after = cli.batch(str(cf))["summary"]
        entry["distillation"] = {**{k: inst[k] for k in ("teacher_heldout", "student_heldout", "params_before", "params_after", "installed")}, "bytes_after": inst.get("bytes_after"),
                                 "reason": inst.get("reason")}
        entry["after"] = {"accuracy": after["accuracy"], "latency_us_p50": after["latency_us_p50"], "bundled_regression": cli.regression(cap)["bundled_test_accuracy"],
                          "version": next(c for c in cli.list() if c["capability_id"] == cap)["active_version"]}
        comp.append(entry)
    out["compaction"] = comp
    # rollback test on an isolated copy of the runtime state (the real state keeps only what passed the gates)
    shutil.copytree(RUN / "rt_cons", RUN / "rt_cons_copy")
    cp = Cli(RUN / "rt_cons_copy", trust)
    for entry in comp:
        cap = entry["capability"]
        if not entry["distillation"]["installed"]:
            entry["rollback_check"] = {"skipped": "nothing was installed"}; continue
        cp.json("rollback", cap)
        cf = RUN / f"eval_{cap}.jsonl"
        entry["rollback_check"] = {"active_version": next(c for c in cp.list() if c["capability_id"] == cap)["active_version"], "accuracy": cp.batch(str(cf))["summary"]["accuracy"],
                                   "original_accuracy": entry["before"]["accuracy"]}

    # --- route optimisation
    plan = P.count_inside("id"); dup = json.loads(json.dumps(plan))
    extra = [dict(n, id=n["id"] + "_dup", out=n["out"] + "_dup") for n in plan["nodes"] if n["op"] == "cap"]
    dup["nodes"] = plan["nodes"][:4] + extra + plan["nodes"][4:]; dup["id"] = "count_inside_redundant"
    redundant_sort = P.sort4_network("id")
    nn = []
    for n in redundant_sort["nodes"]:
        if n["op"] == "cap":
            nn.append(dict(n, id=n["id"] + "_x", out=n["out"] + "_x")); nn.append(n)
        else:
            if n["op"] == "cond_swap":
                n = dict(n, flag={"slot": n["flag"]["slot"] + "_x"})
            nn.append(n)
    redundant_sort["nodes"] = nn; redundant_sort["id"] = "sort4_redundant"
    opts = []
    for name, pl, gen in (("count_inside", dup, lambda r: {"p": [r.uniform(-1, 1) for _ in range(8)]}),
                          ("sort4", redundant_sort, lambda r: {"v": [k / 19 for k in r.sample(range(20), 4)]})):
        op, st = optimise(pl); rr = random.Random(9); cases = [gen(rr) for _ in range(300)]
        cf = RUN / f"opt_cases_{name}.jsonl"; cf.write_text("\n".join(json.dumps({"inputs": c}) for c in cases) + "\n")
        res = {}
        for tag, plan_ in (("original", pl), ("optimised", op)):
            pf = RUN / f"opt_{name}_{tag}.json"; pf.write_text(json.dumps(plan_)); res[tag] = cli.plan_batch(str(pf), str(cf))
        same = all(a["outputs"] == b["outputs"] for a, b in zip(res["original"]["results"], res["optimised"]["results"]))
        opts.append({"plan": name, **st, "outputs_identical": same, "latency_us_p50_before": res["original"]["summary"]["total_us_p50"], "latency_us_p50_after": res["optimised"]["summary"]["total_us_p50"]})
    out["route_optimisation"] = opts
    out["after"] = total_stats(cli)
    out["registry_after"] = [{k: c[k] for k in ("capability_id", "archived", "active_version", "params", "model_bytes", "calls")} for c in cli.json("list-all")]
    return out


# ---------------------------------------------------------------- B. learned routing
def router_eval_sets(rng):
    inscope = {"compare_numbers": ["comparison relation of numbers", "which of these numbers is bigger", "order the two values", "is the first one greater than, smaller than or the same as the second",
                                   "is the second figure bigger than the first", "rank these two quantities", "which of the pair is larger", "do these two numbers match or differ in size", CANON["compare_numbers"]],
               "point_region": ["geometry check point inside circle", "does this coordinate fall within the disc", "classify the point as inside or outside", "locate point relative to the circular region",
                                "is the dot inside the ring", "does the coordinate pair sit in the round zone", "is this spot within the circle boundary", "tell me whether the point is outside the circle", CANON["point_region"]],
               "argmax_position": ["which index holds the maximum", "where is the biggest entry", "pick the winning slot", "position of the highest value",
                                   "which element is the greatest", "find where the peak value is", "identify the largest entry's index", "point to the biggest of the four numbers", CANON["argmax_position"]]}
    oos2 = OOS[2] + ["book me a table for two", "how do I bake bread", "what is the capital of peru", "play a relaxing playlist"]
    oos4 = OOS[4] + ["remind me to call mom", "tell me a joke", "draw a cat", "explain recursion"]
    adv2 = ADV[2] + ["compare the two job offers", "check whether the point of the argument holds", "inside the box there is a cat"]
    adv4 = ADV[4] + ["find the largest planet", "region of the brain responsible for memory", "position of the sun at noon"]
    rows = []
    for cap, intents in inscope.items():
        for it in intents:
            rows.append({"set": "in_scope", "gold": cap, "intent": it, "input": inputs_for(cap, rng)[0]})
    for dim, lst, st in ((2, oos2, "out_of_scope"), (4, oos4, "out_of_scope"), (2, adv2, "adversarial"), (4, adv4, "adversarial")):
        for it in lst:
            rows.append({"set": st, "gold": UNKNOWN, "intent": it, "input": [round(rng.random(), 4) for _ in range(dim)]})
    return rows


def eval_router(cli, rows, router, tag):
    f = RUN / f"router_eval_{tag}.jsonl"
    f.write_text("\n".join(json.dumps({"intent": r["intent"], "input": r["input"]}) for r in rows) + "\n")
    res = cli._run_detect("full", "batch", "--cases", str(f), router=router)["results"]
    m = {"in_scope": {"n": 0, "correct_known": 0, "routed_correct": 0, "misrouted_known": 0, "needs_help": 0},
         "out_of_scope": {"n": 0, "fabrication_known": 0, "uncertain": 0, "rejected": 0}, "adversarial": {"n": 0, "fabrication_known": 0, "uncertain": 0, "rejected": 0}}
    for r, x in zip(rows, res):
        d = m[r["set"]]; d["n"] += 1
        if r["set"] == "in_scope":
            if x.get("needs_help"): d["needs_help"] += 1
            else:
                ok = x["capability"] == r["gold"]
                d["routed_correct"] += ok
                if x["status"] == "KNOWN": d["correct_known"] += ok; d["misrouted_known"] += (not ok)
        else:
            if x.get("needs_help"): d["rejected"] += 1
            elif x["status"] == "KNOWN": d["fabrication_known"] += 1
            else: d["uncertain"] += 1
    for k, d in m.items():
        for key in [k2 for k2 in d if k2 != "n"]:
            d[key] = d[key] / d["n"] if d["n"] else None
    return m


def install_router(cli, svc, caps, hard, version, tag):
    classes = caps + [UNKNOWN]
    t0 = time.time()
    data = make_dataset(caps, seed=0, hard_negatives=hard)
    model, rep = train_router(data, classes)
    onnx_bytes = export_onnx(model, DIM)
    rng = random.Random(0); val = make_dataset(caps, seed=77, n_per_class=20, hard_negatives=hard)
    tests = [(hash_features(t), classes.index(c)) for t, c in val]
    path = RUN / f"router_{tag}_{version}.cap"
    build_cap(path, capability_id="__router__", version=version, model_bytes=onnx_bytes, params=rep["params"], input_dim=DIM, labels=classes, tests=tests,
              keywords=["router"], description="learned intent router", signer_id="build-svc-1", signer_key=svc.key, provenance={"source": "router-trainer", "hard_negatives": hard},
              min_accuracy=0.8, role="router")
    imp = cli.import_caps(str(path))[0]; assert imp["activated"], imp
    rep.update(total_seconds=time.time() - t0, bytes=len(onnx_bytes), val_bundled_accuracy=imp["test_accuracy"])
    return rep


def routing():
    out = {}
    teacher = TeacherSimulator(); svc = BuildService(RUN / "keys", "build-svc-1", RUN / "build_rt", teacher)
    cli = Cli(RUN / "rt_router", RUN / "trust.json"); esc = Escalator(cli, teacher, svc, RUN / "esc_rt")
    caps = ["compare_numbers", "point_region", "argmax_position"]
    for t in caps:
        learn(t, esc)
    rows = router_eval_sets(random.Random(11))
    leaks = overlaps([r["intent"] for r in rows], caps + ["majority_vote"])
    assert not leaks, f"train/eval leak: {leaks}"
    out["eval_rows"] = len(rows); out["eval_train_overlap"] = leaks
    out["keyword_router"] = eval_router(cli, rows, None, "keyword")
    variants = {}
    for hard in (False, True):
        tag = "hard" if hard else "basic"
        rep = install_router(cli, svc, caps, hard, "0.1.0" if not hard else "0.2.0", tag)
        variants[tag] = {"training": {k: rep[k] for k in ("train_seconds", "total_seconds", "val_accuracy", "params", "bytes", "val_bundled_accuracy")}, "eval": eval_router(cli, rows, "learned", f"learned_{tag}")}
    out["learned_router"] = variants
    # latency/memory: repeated in-scope intents
    rep_rows = [r for r in rows if r["set"] == "in_scope"] * 400
    f = RUN / "router_speed.jsonl"; f.write_text("\n".join(json.dumps({"intent": r["intent"], "input": r["input"]}) for r in rep_rows) + "\n")
    speed = {}
    for name, rname in (("keyword", None), ("learned", "learned")):
        s = cli._run_detect("full", "batch", "--cases", str(f), router=rname)["summary"]
        speed[name] = {"cases": s["cases"], "wall_ms": s["wall_ms"], "us_per_task_wall": 1000 * s["wall_ms"] / s["cases"], "peak_rss_kb": s["peak_rss_kb"]}
    out["speed"] = speed
    # staleness: a new capability arrives after the router was trained
    learn("majority_vote", esc)
    maj = [("majority", "decide whether most of the values exceed one half"), ("majority", "does most of the data sit above half"), ("majority", "majority check on five values"),
           ("majority", "which of these five numbers are mostly high")]
    mrows = [{"set": "in_scope", "gold": "majority_vote", "intent": it, "input": [round(random.Random(i).random(), 3) for _ in range(5)]} for i, (_, it) in enumerate(maj)]
    out["new_capability"] = {"keyword_immediately": eval_router(cli, mrows, None, "maj_kw")["in_scope"], "learned_stale": eval_router(cli, mrows, "learned", "maj_stale")["in_scope"]}
    rep = install_router(cli, svc, caps + ["majority_vote"], True, "0.3.0", "retrained")
    out["new_capability"]["learned_after_retrain"] = eval_router(cli, mrows, "learned", "maj_retrained")["in_scope"]
    out["new_capability"]["retrain_seconds"] = rep["total_seconds"]
    out["new_capability"]["keyword_router_retrain_seconds"] = 0.0
    return out


def main():
    if RUN.exists():
        shutil.rmtree(RUN)
    RUN.mkdir(parents=True)
    keygen("build-svc-1", RUN / "keys"); trust = RUN / "trust.json"
    write_trust(trust, {"build-svc-1": (RUN / "keys/build-svc-1.public").read_text()})
    report = {"benchmark_format": "bench/1", "experiment": "phase7-consolidation-and-routing", "platform": platform.platform(), "date": time.strftime("%Y-%m-%d")}
    report["consolidation"] = consolidation(trust)
    c = report["consolidation"]
    print("consolidation before/after:", c["before"], c["after"]); print("pairwise:", c["pairwise_functional_agreement"]); print("merge:", c["merge_duplicate"]["merged"], c["archive_guard"]["refused"], c["archive_unused"])
    report["routing"] = routing()
    r = report["routing"]
    print("keyword:", r["keyword_router"]); print("learned:", {k: v["eval"] for k, v in r["learned_router"].items()}); print("speed:", r["speed"]); print("new cap:", r["new_capability"])
    out = ROOT / "benchmarks/reports/phase7.json"; out.write_text(json.dumps(report, indent=1, default=str)); print("wrote", out)


if __name__ == "__main__":
    main()
