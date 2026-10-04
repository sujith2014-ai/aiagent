"""Phase 14: long-running continual-learning benchmark. One seeded 723-arrival stream over a 36-capability world (introductions, paraphrases, unsupported requests, three rule drifts) replayed against the modular
system and several ablations, an always-ask-the-teacher baseline and single-network baselines; plus module growth, consolidation, routing at scale and seed variance. Everything uses the teacher SIMULATOR (no live LLM)."""
from __future__ import annotations
import collections, json, shutil, sys, time, platform, random, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from scripts.cli import ROOT
from scripts.phase14_lab import World, make_stream, ModularRun, Monolith, BankTeacher, make_help_request, NOUN_SYN, SYN, OOS

RUN = ROOT / "runs" / "phase14"
CK = 60
ARMS = {"base": {}, "serve_uncertain": {"uncertain": "serve"}, "no_frame_filter": {"frame_filter": False}, "cache_refusals": {"cache_refusals": True}, "no_router_update": {"router_updates": False},
        "no_monitor": {"monitor": False}, "feedback_0.1": {"fq": 0.1}, "feedback_1.0": {"fq": 1.0}}


def window_stats(rows, lo, hi):
    w = [r for r in rows if lo <= r["t"] < hi and r["type"] != "drift"]
    cap = [r for r in w if r["cap"]]
    served = [r for r in cap if r["served"]]
    return {"arrivals": len(w), "teacher_calls": sum(r["teacher_calls"] for r in w), "capability_arrivals": len(cap), "served": len(served), "wrong": sum(1 for r in served if r["correct"] is False),
            "answered_without_teacher": sum(1 for r in served if r["teacher_calls"] == 0)}


def summarize(run: ModularRun, stream, wall):
    rows = run.rows; n = max(e["t"] for e in stream) + 1
    cap_rows = [r for r in rows if r["cap"]]; served = [r for r in cap_rows if r["served"]]
    wrong = [r for r in served if r["correct"] is False]
    acts = collections.Counter(run.teacher.by_action)
    new_ok = sum(1 for r in rows if r["type"] == "new" and r["served"] and r["learned"]); new_all = sum(1 for r in rows if r["type"] == "new")
    cks = run.checkpoints
    # retention: accuracy at the first checkpoint after each capability appeared vs the last checkpoint (drift-repaired capabilities are listed separately)
    first = {}
    for c in cks:
        for cap, a in c["accuracy"].items(): first.setdefault(cap, (c["t"], a))
    last = cks[-1]["accuracy"] if cks else {}
    drifted = set(run.drift_times)
    stable = [cap for cap in last if cap in first and cap not in drifted]
    bwt = float(np.mean([last[c] - first[c][1] for c in stable])) if stable else None
    unlearned = [e["cap"] for e in stream if e["type"] == "new" and e["cap"] not in last]
    wasted = acts["cannot_help"] + acts["request_tool"] + acts["use_memory"]
    return {"wall_seconds": round(wall, 1), "arrivals": n, "teacher_calls": run.teacher.calls, "teacher_calls_by_action": dict(acts), "teacher_bytes_in": run.teacher.bytes_in, "teacher_bytes_out": run.teacher.bytes_out,
            "teacher_calls_per_100_arrivals_by_window": [round(100 * window_stats(rows, lo, lo + 100)["teacher_calls"] / max(1, window_stats(rows, lo, lo + 100)["arrivals"]), 1) for lo in range(0, n, 100)],
            "windows_of_100": [window_stats(rows, lo, lo + 100) for lo in range(0, n, 100)],
            "calls_not_leading_to_learning_or_routing": wasted, "cache_hits": run.esc.cache_hits, "capability_arrivals": len(cap_rows), "served": len(served), "unserved_capability_arrivals": len(cap_rows) - len(served),
            "silent_wrong_answers": len(wrong), "silent_wrong_rate": round(len(wrong) / max(1, len(served)), 4),
            "wrong_by_status": dict(collections.Counter(r["status"] for r in wrong)), "wrong_by_type": dict(collections.Counter(r["type"] for r in wrong)),
            "new_capabilities_requested": new_all, "new_capabilities_learned_at_first_request": new_ok, "capabilities_never_learned": sorted(set(unlearned)),
            "modules_final": run.stats(), "checkpoints": [{k: c[k] for k in ("t", "mean_accuracy", "min_accuracy", "active_modules", "params", "model_bytes", "teacher_calls")} for c in cks],
            "final_accuracy_mean": cks[-1]["mean_accuracy"] if cks else None, "final_accuracy_min": cks[-1]["min_accuracy"] if cks else None,
            "backward_transfer_stable_capabilities": bwt, "build_seconds": round(run.build_seconds, 1),
            "drift": {cap: {"drift_t": t, "repairs": [r for r in run.repairs if r["cap"] == cap]} for cap, t in run.drift_times.items()},
            "spurious_repairs": [r for r in run.repairs if r["spurious"]], "router_updates_accepted": run.esc.router_updates,
            "wrong_after_drift_before_repair": {cap: sum(1 for r in wrong if r["cap"] == cap and r["t"] >= t and (not [x for x in run.repairs if x["cap"] == cap] or r["t"] < min(x["t"] for x in run.repairs if x["cap"] == cap)))
                                                for cap, t in run.drift_times.items()}}


def latency_probe(run: ModularRun, world: World) -> dict:
    caps = [c["capability_id"] for c in run.cli.list() if c["capability_id"] in world.by_id]
    rng = random.Random(1); rows = []
    for _ in range(300):
        cap = rng.choice(caps); d = world.input_dim(cap); lo = -1.0 if world.by_id[cap].family in ("zone", "side") else 0.0
        rows.append({"intent": world.canonical(cap), "input": [round(rng.uniform(lo, 1), 3) for _ in range(d)]})
    f = run.tmp / "lat.jsonl"; f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = run.cli._run_detect("full", "batch", "--cases", str(f), "--no-stats")["summary"] if False else run.cli._run_detect("full", "batch", "--cases", str(f))["summary"]
    return {"modules": len(caps), "latency_us_p50": s["latency_us_p50"], "latency_us_p95": s.get("latency_us_p95"), "us_per_task_wall": round(1000 * s["wall_ms"] / s["cases"], 1)}


def run_arm(name: str, opts: dict, seed=0, post=None):
    out = RUN / "arms" / f"{name}_s{seed}.json"
    if out.exists(): return json.loads(out.read_text())
    w = World(seed); fq = opts.get("fq", 0.3); stream = make_stream(w, seed=seed, feedback_q=fq)
    run = ModularRun(RUN / f"arm_{name}_s{seed}", w, cache_refusals=opts.get("cache_refusals", False), router_updates=opts.get("router_updates", True), monitor=opts.get("monitor", True), uncertain=opts.get("uncertain", "escalate"))
    run.esc.filter_frame_words = opts.get("frame_filter", True)
    lat = []
    t0 = time.time(); run.run(stream, CK, on_checkpoint=lambda r, t: lat.append({"t": t, **latency_probe(r, w)}))
    rep = summarize(run, stream, time.time() - t0); rep["latency_by_checkpoint"] = lat; rep["options"] = opts; rep["seed"] = seed
    if post: rep["post"] = post(run, w, stream)
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rep, indent=1, default=str))
    rep["_repairs_for_baselines"] = [(r["t"], r["cap"]) for r in run.repairs if r["result"] == "ANSWER"]
    return rep


# ------------------------------------------------------------------------------------------ post-run analyses on the live base run
def routing_at_scale(run: ModularRun, w: World) -> dict:
    from training.routing import train_router, hash_features, DIM as RDIM
    from training.exporters.onnx_export import export_onnx
    from packages.capbuild import build_cap
    caps = [c["capability_id"] for c in run.cli.list() if c["capability_id"] in w.by_id]
    classes = caps + ["UNKNOWN"]
    para = {c: w.paraphrases(c) for c in caps}                       # [easy1, easy2, hard1, hard2]
    train = [(w.canonical(c), c) for c in caps] + [(para[c][0], c) for c in caps] + [(para[c][2], c) for c in caps]      # canonical + one easy + one hard form
    unk_train = [t for k in ("cannot_help", "request_tool") for t in OOS[k]][:5] + ["tell me a story", "what time is it", "play music"]
    train += [(t, "UNKNOWN") for t in unk_train for _ in range(6)]
    rng = random.Random(0); train += [(w.canonical(c).replace("check", "inspect"), c) for c in caps]                    # light augmentation
    model, rep = train_router(train, classes, epochs=200)
    val = [(t, c) for c in caps for t in (w.canonical(c), para[c][0])][:60]
    tests = [(hash_features(t), classes.index(c)) for t, c in val]
    p = run.tmp / "router36.cap"
    build_cap(p, capability_id="__router__", version="0.1.0", model_bytes=export_onnx(model, RDIM), params=rep["params"], input_dim=RDIM, labels=classes, tests=tests, keywords=["router"], description="learned router",
              signer_id="build-svc-1", signer_key=run.svc.key, provenance={"source": "phase14"}, min_accuracy=0.8, role="router")
    imp = run.cli.import_caps(str(p))[0]
    rows = []
    for c in caps:
        d = w.input_dim(c); lo = -1.0 if w.by_id[c].family in ("zone", "side") else 0.0; x = [round(rng.uniform(lo, 1), 3) for _ in range(d)]
        for kind, t in (("canonical", w.canonical(c)), ("easy_unseen", para[c][1]), ("hard_unseen", para[c][3])): rows.append({"set": kind, "gold": c, "intent": t, "input": x})
    for t in ["translate this paragraph into french", "write me a poem about the sea", "what is the capital of peru", "explain recursion simply", "book me a flight", "draw a cat", "sort my emails", "how do i bake bread"]:
        for d in (1, 2, 4): rows.append({"set": "out_of_scope", "gold": "UNKNOWN", "intent": t, "input": [round(rng.random(), 3) for _ in range(d)]})
    out = {"modules": len(caps), "learned_router": {"val_accuracy": rep["val_accuracy"], "train_seconds": round(rep["train_seconds"], 2), "params": rep["params"], "installed": imp["activated"]}}
    for rname, router in (("keyword_with_aliases_learned_in_stream", None), ("learned", "learned")):
        f = run.tmp / f"r_{rname}.jsonl"; f.write_text("\n".join(json.dumps({"intent": r["intent"], "input": r["input"]}) for r in rows) + "\n")
        res = run.cli._run_detect("full", "batch", "--cases", str(f), router=router)["results"]
        m = collections.defaultdict(lambda: {"n": 0, "correct_known": 0, "wrong_known": 0, "needs_help": 0, "other": 0})
        for r, x in zip(rows, res):
            d = m[r["set"]]; d["n"] += 1
            if x.get("needs_help"): d["needs_help"] += 1
            elif r["set"] == "out_of_scope": d["wrong_known"] += 1 if x["status"] == "KNOWN" else 0; d["other"] += 0 if x["status"] == "KNOWN" else 1
            elif x["status"] == "KNOWN": d["correct_known" if x["capability"] == r["gold"] else "wrong_known"] += 1
            else: d["other"] += 1
        out[rname] = {k: {kk: (vv / v["n"] if kk != "n" else vv) for kk, vv in v.items()} for k, v in m.items()}
    return out


def consolidation_analysis(run: ModularRun, w: World) -> dict:
    from training import consolidation as C
    out = {"before": run.stats(), "accuracy_before": float(np.mean(list(run.accuracy_all(9999).values())))}
    rows = C.pairwise_report(run.cli, run.tmp)
    out["pairs_compared"] = len(rows); out["pairs_flagged"] = [r for r in rows if r["verdict"] != "different"]
    twins = [(c.twin_of, c.cap_id) for c in w.caps if c.twin_of]
    out["twin_pairs"] = []
    suite = list(run.esc.routing_suite)[:80]
    for keeper, dup in twins:
        r = next((x for x in rows if {x["a"], x["b"]} == {keeper, dup}), None)
        entry = {"keeper": keeper, "duplicate": dup, "agreement": r["agreement_in_distribution"] if r else None, "verdict": r["verdict"] if r else "not_comparable",
                 "rules_identical_in_world": w.by_id[keeper].params == w.by_id[dup].params}
        if r and r["verdict"] == "duplicate":
            entry["merge"] = C.merge_duplicate(run.cli, run.svc, keeper, dup, suite, run.tmp)
        out["twin_pairs"].append(entry)
    out["after_merge"] = run.stats(); acc = run.accuracy_all(10000)
    out["accuracy_after_merge"] = float(np.mean(list(acc.values())))
    # compaction: distill every active module to (16,16) under the 0.01 held-out gate
    comp = {"attempted": 0, "installed": 0, "rejected": 0, "params_before": 0, "params_after": 0, "teacher_heldout_mean": [], "student_heldout_mean": []}
    p0 = run.stats()["params"]
    for c in run.cli.list():
        cap = c["capability_id"]
        if cap not in run.svc.models or cap not in w.by_id: continue
        comp["attempted"] += 1
        stud, rep = C.distill(run.svc, cap, hidden=(16, 16)); res = C.compact(run.svc, run.cli, cap, stud, rep)
        comp["installed" if res["installed"] else "rejected"] += 1; comp["teacher_heldout_mean"].append(rep["teacher_heldout"]); comp["student_heldout_mean"].append(rep["student_heldout"])
    comp["teacher_heldout_mean"] = float(np.mean(comp["teacher_heldout_mean"])); comp["student_heldout_mean"] = float(np.mean(comp["student_heldout_mean"]))
    out["compaction"] = comp; out["after_compaction"] = run.stats(); out["params_before_compaction"] = p0
    acc2 = run.accuracy_all(10001); out["accuracy_after_compaction"] = float(np.mean(list(acc2.values())))
    # archive-by-usage cost (analytic: what share of traffic the archived capabilities would have been serving)
    calls = {c["capability_id"]: c["calls"] for c in run.cli.json("list-all") if c["role"] == "capability"}
    tot = sum(calls.values()) or 1
    out["archive_policy"] = {f"fewer_than_{k}_calls": {"modules": sum(1 for v in calls.values() if v < k), "share_of_traffic": round(sum(v for v in calls.values() if v < k) / tot, 4)} for k in (2, 3, 5, 10)}
    return out


def post_base(run, w, stream):
    return {"routing_at_scale": routing_at_scale(run, w), "consolidation": consolidation_analysis(run, w)}


# ------------------------------------------------------------------------------------------ baselines replayed on the same stream
def baselines(base_rep: dict, seed=0) -> dict:
    out = {}
    stream_w = World(seed); stream = make_stream(stream_w, seed=seed)
    repairs = {t: cap for t, cap in base_rep["_repairs_for_baselines"]}                  # baselines are repaired when the modular system repaired (same detection delay)
    ck_times = [c["t"] for c in base_rep["checkpoints"]]
    for mode in ("finetune", "replay", "joint"):
        w = World(seed); m = Monolith([c.cap_id for c in w.caps], mode, seed=seed); introduced = []; cks = []; first = {}
        for ev in stream:
            t = ev["t"]
            if ev["type"] == "drift": w.drift(ev["cap"])
            if ev["type"] == "new":
                xs, ys = w.sample(ev["cap"], 800, 100 + t); m.learn(ev["cap"], xs, ys); introduced.append(ev["cap"])
            if t in repairs:
                xs, ys = w.sample(repairs[t], 800, 7000 + t); m.learn(repairs[t], xs, ys)
            if t in ck_times:
                acc = {c: m.accuracy(c, *w.sample(c, 100, 1000 + t)) for c in introduced}
                for c, a in acc.items(): first.setdefault(c, a)
                cks.append({"t": t, "mean_accuracy": float(np.mean(list(acc.values()))), "min_accuracy": min(acc.values())})
        last = {c: m.accuracy(c, *w.sample(c, 100, 1000 + ck_times[-1])) for c in introduced}
        drifted = {ev["cap"] for ev in stream if ev["type"] == "drift"}
        out[mode] = {"checkpoints": cks, "final_accuracy_mean": cks[-1]["mean_accuracy"], "final_accuracy_min": cks[-1]["min_accuracy"], "params": m.params(), "train_seconds": round(m.train_seconds, 1), "learn_events": m.events,
                     "backward_transfer_stable_capabilities": float(np.mean([last[c] - first[c] for c in introduced if c not in drifted])), "teacher_calls": sum(1 for ev in stream if ev["type"] == "new") + len(repairs),
                     "note": "oracle task routing and oracle capability identity (the modular system does not get these)"}
    return out


def teacher_always(seed=0) -> dict:
    w = World(seed); stream = make_stream(w, seed=seed); t = BankTeacher(w); introduced = []; calls = 0; bytes_in = 0
    for ev in stream:
        if ev["type"] == "drift": continue
        if ev["type"] == "new": introduced.append(ev["cap"])
        dim = len(ev["x"]); req = make_help_request(ev["intent"], dim, introduced, {"reason_code": "always_ask"}, "")
        bytes_in += len(req.to_json()); calls += 1
    return {"teacher_calls": calls, "teacher_bytes_in": bytes_in, "arrivals": calls, "note": "every non-drift arrival is sent to the teacher; accuracy equals the teacher's by assumption; no module is ever trained"}


def main():
    if RUN.exists() and "--fresh" in sys.argv: shutil.rmtree(RUN)
    RUN.mkdir(parents=True, exist_ok=True)
    rep = {"benchmark_format": "bench/1", "experiment": "phase14-long-running-continual-learning", "platform": platform.platform(), "date": time.strftime("%Y-%m-%d"),
           "caveats": ["teacher is a simulator that knows the world (including drift); no live LLM", "synthetic rule families, 36 capabilities, one world seed for the ablations",
                       "baselines get oracle routing and borrow the modular system's repair times", "feedback is an oracle that labels a random 30% of served answers"]}
    arms = {}
    for name, opts in ARMS.items():
        t0 = time.time(); arms[name] = run_arm(name, opts, 0, post=post_base if name == "base" else None); print(name, "done", round(time.time() - t0), "s", "calls", arms[name]["teacher_calls"], "wrong", arms[name]["silent_wrong_answers"], flush=True)
    rep["arms"] = {k: {kk: vv for kk, vv in v.items() if kk != "_repairs_for_baselines"} for k, v in arms.items()}
    rep["baselines"] = baselines(arms["base"]); rep["teacher_always"] = teacher_always()
    seeds = {}
    for s in (1, 2):
        r = run_arm("base", {}, s); seeds[str(s)] = {k: r[k] for k in ("teacher_calls", "silent_wrong_answers", "silent_wrong_rate", "final_accuracy_mean", "final_accuracy_min", "new_capabilities_learned_at_first_request", "capabilities_never_learned", "modules_final", "build_seconds")}
        seeds[str(s)]["drift_delays"] = {c: [x["delay"] for x in d["repairs"]] for c, d in r["drift"].items()}
        print("seed", s, "done", flush=True)
    b = arms["base"]; seeds["0"] = {k: b[k] for k in ("teacher_calls", "silent_wrong_answers", "silent_wrong_rate", "final_accuracy_mean", "final_accuracy_min", "new_capabilities_learned_at_first_request", "capabilities_never_learned", "modules_final", "build_seconds")}
    seeds["0"]["drift_delays"] = {c: [x["delay"] for x in d["repairs"]] for c, d in b["drift"].items()}
    rep["seeds"] = seeds
    (ROOT / "benchmarks/reports/phase14.json").write_text(json.dumps(rep, indent=1, default=str)); print("ok")


if __name__ == "__main__":
    main()
