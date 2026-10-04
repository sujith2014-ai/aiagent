"""Phase 4: dynamic composition over independently learned modules, with module reuse, branching and loops,
routing overhead (intent vs id), error accumulation, and end-to-end tiny-network baselines."""
from __future__ import annotations
import json, random, shutil, sys, time, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch, torch.nn as nn
from scripts.cli import Cli, ROOT
from scripts import plans as P
from packages.capbuild import keygen, write_trust
from integrations.teacher.simulator import TeacherSimulator
from integrations.teacher.provider import HelpRequest
from training.service import BuildService
from training.trainer.train import make_mlp, count_params
from training.exporters.onnx_export import export_onnx
from packages.capbuild import build_cap

RUN = ROOT / "runs" / "phase4"
N_EVAL, N_TRAIN_BASE = 1000, 3000
G = 19


def rv(rng):  # 4 distinct grid values
    return [k / G for k in rng.sample(range(20), 4)]


def rp(rng, n):
    return [rng.uniform(-1, 1) for _ in range(n)]


def truth_region(x, y): return 0 if x * x + y * y < 0.5 else 1
def truth_cmp(a, b): return 0 if a < b else 1 if abs(a - b) < 1e-9 else 2
def truth_arg(v): return max(range(4), key=lambda i: v[i])


# each composite: sample() -> (inputs dict, flat feature vector for baseline, truth outputs list)
def s_sort(rng):
    v = rv(rng); return {"v": v}, v, sorted(v)
def s_count(rng):
    p = rp(rng, 8); return {"p": p}, p, [sum(truth_region(p[2*k], p[2*k+1]) == 0 for k in range(4))]
def s_mixed(rng):
    v, w = rv(rng), rp(rng, 4); i = truth_arg(v)
    return {"v": v, "w": w}, v + w, [truth_region(w[i], w[(i+1) % 4]), truth_cmp(v[i], 10/19)]
def s_branch(rng):
    a = [rng.randrange(20)/G, rng.randrange(20)/G]; p = rp(rng, 2); v = rv(rng)
    r = truth_cmp(*a); o = truth_region(*p) if r == 2 else truth_arg(v)
    return {"a": a, "p": p, "v": v}, a + p + v, [r, o]
def s_path(rng):
    a = [rng.randrange(20)/G, rng.randrange(20)/G]; a2 = [rng.randrange(20)/G, rng.randrange(20)/G]; p = rp(rng, 2); v = rv(rng)
    return {"a": a, "a2": a2, "p": p, "v": v}, a + a2 + p + v, [truth_cmp(*a), truth_region(*p), truth_cmp(*a2), truth_arg(v)]


def perm_class(vals):  # baseline label for sort: argsort permutation index
    import itertools
    order = tuple(sorted(range(4), key=lambda i: vals[i]))
    return list(itertools.permutations(range(4))).index(order)


COMPOSITES = {
    "sort4": dict(sample=s_sort, plans={"network": P.sort4_network, "bubble_loop": P.sort4_bubble}, classes=24,
                  base_label=lambda x, t: perm_class(x), same=lambda out, t: all(abs(a - b) < 1e-5 for a, b in zip(out, t))),
    "count_inside": dict(sample=s_count, plans={"main": P.count_inside}, classes=5,
                  base_label=lambda x, t: int(t[0]), same=lambda out, t: int(round(out[0])) == int(t[0])),
    "mixed_C_select_B_A": dict(sample=s_mixed, plans={"main": P.mixed}, classes=6,
                  base_label=lambda x, t: t[0]*3 + t[1], same=lambda out, t: [int(round(o)) for o in out] == t),
    "branch": dict(sample=s_branch, plans={"main": P.branch}, classes=12,
                  base_label=lambda x, t: t[0]*4 + t[1], same=lambda out, t: [int(round(o)) for o in out] == t),
    "path_ABAC": dict(sample=s_path, plans={"main": P.path_abac}, classes=3*2*3*4,
                  base_label=lambda x, t: ((t[0]*2 + t[1])*3 + t[2])*4 + t[3], same=lambda out, t: [int(round(o)) for o in out] == t),
}


def baseline(name, spec, seeds=(0, 1, 2)):
    rng = random.Random(555)
    tr = [spec["sample"](rng) for _ in range(N_TRAIN_BASE)]
    te = [spec["sample"](random.Random(999 + k)) for k in range(N_EVAL)]
    xt = torch.tensor([t[1] for t in tr], dtype=torch.float32); yt = torch.tensor([spec["base_label"](t[1], t[2]) for t in tr])
    xe = torch.tensor([t[1] for t in te], dtype=torch.float32); ye = torch.tensor([spec["base_label"](t[1], t[2]) for t in te])
    accs, params, last = [], None, None
    for sd in seeds:
        torch.manual_seed(sd)
        m = make_mlp(xt.shape[1], spec["classes"], (64, 64)); params = count_params(m)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        for ep in range(150):
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 128):
                idx = perm[i:i+128]; opt.zero_grad(); nn.functional.cross_entropy(m(xt[idx]), yt[idx]).backward(); opt.step()
        with torch.no_grad():
            accs.append((m(xe).argmax(1) == ye).float().mean().item())
        last = m
    baseline.last_model, baseline.last_eval = last, (xe, ye)
    return {"train_examples": N_TRAIN_BASE, "hidden": [64, 64], "params": params, "accuracy_seeds": accs, "accuracy_mean": sum(accs) / len(accs)}


def main():
    if RUN.exists():
        shutil.rmtree(RUN)
    RUN.mkdir(parents=True)
    keygen("build-svc-1", RUN / "keys")
    trust = RUN / "trust.json"; write_trust(trust, {"build-svc-1": (RUN / "keys/build-svc-1.public").read_text()})
    teacher = TeacherSimulator(); svc = BuildService(RUN / "keys", "build-svc-1", RUN / "build", teacher)
    rt = Cli(RUN / "rt", trust); known = set(); module_acc = {}
    for intent, dim in [("compare numbers relation", 2), ("point inside region", 2), ("largest value position", 4)]:
        lp = teacher.respond(HelpRequest(intent, dim)); b = svc.build(lp, "0.1.0", known); assert b.promoted
        assert rt.import_caps(str(b.cap_path))[0]["activated"]; known.add(lp.capability_id)
        module_acc[lp.capability_id] = b.report["attempts"][-1]["heldout_accuracy"]
    report = {"benchmark_format": "bench/1", "experiment": "phase4-composition", "platform": platform.platform(),
              "date": time.strftime("%Y-%m-%d"), "module_heldout_accuracy": module_acc, "composites": {}}
    for name, spec in COMPOSITES.items():
        rng = random.Random(31337)
        cases = [spec["sample"](rng) for _ in range(N_EVAL)]
        cf = RUN / f"cases_{name}.jsonl"
        cf.write_text("\n".join(json.dumps({"inputs": c[0]}) for c in cases) + "\n")
        entry = {"variants": {}}
        for vname, mk in spec["plans"].items():
            for by in ("intent", "id"):
                plan = mk(by); pf = RUN / f"{plan['id']}.json"; pf.write_text(json.dumps(plan))
                v = rt.plan_validate(str(pf)); assert v["valid"], v
                out = rt.plan_batch(str(pf), str(cf))
                ok = sum(1 for r, c in zip(out["results"], cases) if "outputs" in r and spec["same"](r["outputs"], c[2]))
                calls = {}
                for r in out["results"]:
                    for k, n in r.get("calls", {}).items(): calls[k] = max(calls.get(k, 0), n)
                entry["variants"][f"{vname}/{by}"] = {
                    "accuracy": ok / len(cases), "needs_help": out["summary"]["needs_help"], "errors": sum(1 for r in out["results"] if "error" in r),
                    "node_executions_p50": out["summary"]["node_executions_p50"], "capability_calls_per_task": calls,
                    "latency_us_p50": {"total": out["summary"]["total_us_p50"], "modules": out["summary"]["module_us_p50"], "overhead": out["summary"]["overhead_us_p50"]},
                    "parallel_levels_example": out["results"][0].get("levels"), "peak_rss_kb": out["summary"]["peak_rss_kb"]}
                if name == "branch" and by == "intent":
                    # exactly one branch capability may execute per task (no eager evaluation of the other branch)
                    branch_caps = [sum(1 for x in r["records"] if x["node"] in ("tb", "ec")) for r in out["results"]]
                    entry["branch_exclusive"] = all(n == 1 for n in branch_caps)
                    entry["branch_taken_then_fraction"] = sum(1 for r in out["results"] if any(x["node"] == "tb" for x in r["records"])) / len(cases)
                if name == "path_ABAC" and by == "intent":
                    entry["path_order_ok"] = all([x["cap"] for x in r["records"]] == ["compare_numbers", "point_region", "compare_numbers", "argmax_position"] for r in out["results"])
                if vname == list(spec["plans"])[0] and by == "intent":
                    entry["example_trace"] = out["results"][0]["records"][:8]
        entry["baseline_end_to_end_mlp"] = baseline(name, spec)
        report["composites"][name] = entry
        best = max(v["accuracy"] for v in entry["variants"].values())
        print(f"[{name}] modular={ {k: round(v['accuracy'], 3) for k, v in entry['variants'].items()} } baseline={entry['baseline_end_to_end_mlp']['accuracy_mean']:.3f}")
    # latency of the monolithic end-to-end network through the *same* backend (count_inside, mixed)
    mono = {}
    rtm = Cli(RUN / "rt-mono", trust)
    for name, kws in [("count_inside", ["monolithic", "count"]), ("mixed_C_select_B_A", ["monolithic", "mixed"])]:
        spec = COMPOSITES[name]; baseline(name, spec, seeds=(0,))
        m = baseline.last_model; xe, ye = baseline.last_eval; dim = xe.shape[1]
        onnx_bytes = export_onnx(m, dim)
        tests = [(xe[i].tolist(), int(ye[i])) for i in range(100)]
        cap_path = RUN / f"mono_{name}.cap"
        build_cap(cap_path, capability_id=f"mono_{name}", version="0.1.0", model_bytes=onnx_bytes, params=count_params(m), input_dim=dim,
                  labels=[str(i) for i in range(spec["classes"])], tests=tests, keywords=kws, description="monolithic baseline", signer_id="build-svc-1",
                  signer_key=svc.key, provenance={"source": "baseline"}, min_accuracy=0.3)
        assert rtm.import_caps(str(cap_path))[0]["activated"]
        cf = RUN / f"mono_cases_{name}.jsonl"
        cf.write_text("\n".join(json.dumps({"intent": " ".join(kws), "input": xe[i].tolist()}) for i in range(500)) + "\n")
        b = rtm.batch(str(cf))
        modular = report["composites"][name]["variants"]["main/intent"]["latency_us_p50"]["total"]
        mono[name] = {"monolithic_inference_us_p50": b["summary"]["latency_us_p50"], "modular_plan_total_us_p50": modular,
                      "latency_ratio_modular_over_monolithic": modular / max(1, b["summary"]["latency_us_p50"])}
    report["latency_modular_vs_monolithic"] = mono
    # expected-from-components sanity check (independent-error approximation)
    a, b, c = module_acc["compare_numbers"], module_acc["point_region"], module_acc["argmax_position"]
    report["expected_accuracy_if_errors_independent"] = {"count_inside": b ** 4, "mixed_C_select_B_A": c * b * a, "path_ABAC": a * b * a * c,
                                                       "branch": "see note: depends on branch taken", "sort4": "a**5 (network), a**9 (bubble) upper-bound estimate: " + f"{a**5:.3f} / {a**9:.3f}"}
    # safety: loops and budgets
    bad = {"id": "unbounded", "inputs": {"v": 1}, "nodes": [{"op": "repeat", "id": "r", "times": 10**6, "body": []}], "outputs": [S("v")]} if False else \
          {"id": "unbounded", "inputs": {"v": 1}, "nodes": [{"op": "repeat", "id": "r", "times": 10**6, "body": []}], "outputs": [P.S("v")]}
    (RUN / "bad1.json").write_text(json.dumps(bad))
    nested = {"id": "nested", "inputs": {"v": 1}, "nodes": [{"op": "repeat", "id": "o", "times": 1000, "body": [{"op": "repeat", "id": "i", "times": 1000, "body": [
        {"op": "gather", "id": "g", "args": [P.S("v")], "out": "v"}]}]}], "outputs": [P.S("v")]}
    (RUN / "bad2.json").write_text(json.dumps(nested)); (RUN / "one.jsonl").write_text(json.dumps({"inputs": {"v": [1.0]}}) + "\n")
    v1 = rt.plan_validate(str(RUN / "bad1.json")); v2 = rt.plan_batch(str(RUN / "bad2.json"), str(RUN / "one.jsonl"))
    report["loop_safety"] = {"repeat_1e6_rejected_at_validation": not v1["valid"], "nested_1000x1000_aborted_by_step_budget": "budget" in v2["results"][0].get("error", "")}
    out = ROOT / "benchmarks/reports/phase4.json"; out.write_text(json.dumps(report, indent=1)); print("wrote", out)


if __name__ == "__main__":
    main()
