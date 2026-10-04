"""Phase 2+3 experiment: unknown -> teacher -> LearningPackage -> train -> .cap -> registry -> solve without teacher,
sequentially for 3 capabilities, with regression, clean-restart portability, constrained-device simulation, baselines."""
from __future__ import annotations
import json, shutil, sys, time, os, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.cli import Cli, ROOT
from packages.capbuild import keygen, write_trust
from integrations.teacher.simulator import TeacherSimulator, TASKS, sample, splits
from integrations.teacher.provider import HelpRequest
from training.service import BuildService
from training.evaluation import baselines

RUN = ROOT / "runs" / "phase3"
ORDER = [("compare_numbers", "compare these two numbers"), ("point_region", "is this point inside the circular region"),
         ("argmax_position", "find the position of the largest value")]
PARAPHRASE = {"compare_numbers": "comparison relation of numbers", "point_region": "geometry check point inside circle",
              "argmax_position": "which index holds the maximum"}


def eval_cases(task, path):
    xs, ys = splits(task)["eval"]  # disjoint from train/val/bundled-test (except compare_numbers: eval == test)
    with open(path, "w") as f:
        for x, y in zip(xs, ys):
            f.write(json.dumps({"intent": ORDER_INTENT[task], "input": x, "expected_index": y, "expected_capability": task}) + "\n")
    return xs, ys


ORDER_INTENT = dict(ORDER)


def main():
    if RUN.exists():
        shutil.rmtree(RUN)
    RUN.mkdir(parents=True)
    keys = RUN / "keys"
    keygen("build-svc-1", keys)
    trust = RUN / "trust.json"
    write_trust(trust, {"build-svc-1": (keys / "build-svc-1.public").read_text()})
    teacher = TeacherSimulator()
    svc = BuildService(keys, "build-svc-1", RUN / "build", teacher)
    rt = Cli(RUN / "rt-a", trust)
    stages, teacher_log, report = [], [], {"benchmark_format": "bench/1", "experiment": "phase3-sequential-learning", "platform": platform.platform(), "date": time.strftime("%Y-%m-%d")}
    known = set()
    for stage_i, (task, intent) in enumerate(ORDER):
        ex = sample(task, 1, 99)[0][0]
        first = rt.solve(intent, ex)
        assert first["result"] == "NEEDS_HELP", f"first encounter of {task} should be NEEDS_HELP, got {first}"
        t0 = time.time()
        lp = teacher.respond(HelpRequest(intent, len(ex), sorted(known), {"decision": "unknown"}, "no capability"))
        built = svc.build(lp, "0.1.0", known)
        assert built.promoted, built.reason
        imp = rt.import_caps(str(built.cap_path))[0]
        assert imp["activated"], imp
        learn_seconds = time.time() - t0
        again = rt.solve(intent, ex)
        assert again["result"] == "ANSWER", again
        known.add(task)
        teacher_log.append({"task": task, "teacher_calls_total": teacher.calls})
        # related encounter X': new example, paraphrased intent, no teacher
        calls_before = teacher.calls
        xs2, ys2 = sample(task, 1, 4242)
        rel = rt.solve(PARAPHRASE[task], xs2[0])
        # evaluate all capabilities learned so far on unseen examples
        acc = {}
        for k in sorted(known):
            cf = RUN / f"eval_{k}.jsonl"
            eval_cases(k, cf)
            b = rt.batch(str(cf))
            acc[k] = {"accuracy": b["summary"]["accuracy"], "needs_help": b["summary"]["needs_help"],
                      "latency_us_p50": b["summary"]["latency_us_p50"], "peak_rss_kb": b["summary"]["peak_rss_kb"]}
        regs = {k: rt.regression(k)["bundled_test_accuracy"] for k in sorted(known)}
        stages.append({"learned": task, "teacher_calls_for_stage": 1, "learn_seconds": learn_seconds,
                       "build": built.report, "import_steps": imp["steps"],
                       "related_encounter": {"intent": PARAPHRASE[task], "result": rel["result"], "status": rel.get("status"),
                                             "capability": rel.get("capability_id"), "teacher_called": teacher.calls != calls_before},
                       "accuracy_all_known": acc, "bundled_regression": regs})
        print(f"[{task}] learned in {learn_seconds:.1f}s; accuracies: " + ", ".join(f"{k}={v['accuracy']:.3f}" for k, v in acc.items()))
    report["stages"] = stages
    report["registry"] = rt.list()

    # forgetting: accuracy of each earlier capability at each later stage vs right after it was learned
    forgetting = []
    for i, st in enumerate(stages):
        for k, v in st["accuracy_all_known"].items():
            first_acc = stages[[s["learned"] for s in stages].index(k)]["accuracy_all_known"][k]["accuracy"]
            forgetting.append({"stage": st["learned"], "capability": k, "acc": v["accuracy"], "drop_vs_first": first_acc - v["accuracy"]})
    report["forgetting"] = forgetting
    report["max_forgetting_drop"] = max(f["drop_vs_first"] for f in forgetting)

    # export -> destroy runtime state -> fresh install -> import -> compare outputs
    final_dir = RUN / "export"; final_dir.mkdir()
    for f in (RUN / "rt-a" / "store").glob("*.cap"):
        shutil.copy(f, final_dir / f.name)
    before = {}
    for k, _ in ORDER:
        before[k] = rt.batch(str(RUN / f"eval_{k}.jsonl"))
    shutil.rmtree(RUN / "rt-a")
    rt2 = Cli(RUN / "rt-b", trust)
    reps = rt2.import_caps(*sorted(map(str, final_dir.glob("*.cap"))))
    assert all(r["activated"] for r in reps), reps
    fresh, same = {}, True
    for k, _ in ORDER:
        b = rt2.batch(str(RUN / f"eval_{k}.jsonl"))
        identical = [x["probs"] for x in b["results"]] == [x["probs"] for x in before[k]["results"]]
        same &= identical
        fresh[k] = {"accuracy": b["summary"]["accuracy"], "identical_probs_to_pre_restart": identical}
    report["clean_restart"] = {"fresh_runtime_imports": len(reps), "per_capability": fresh, "all_identical": same}
    print("clean restart identical:", same)

    # constrained-device simulation: same .cap files, single CPU, 1 GiB address-space cap, Android-like profile
    cons_root = RUN / "rt-c"
    prefix = ["bash", "-c", 'ulimit -v 1048576; exec "$@"', "_"]
    try:
        import shutil as sh
        if sh.which("taskset"):
            prefix = ["bash", "-c", 'ulimit -v 1048576; exec taskset -c 0 "$@"', "_"]
    except Exception:
        pass
    rtc = Cli(cons_root, trust, device="PC_CONSTRAINED", prefix=prefix)
    creps = rtc.import_caps(*sorted(map(str, final_dir.glob("*.cap"))))
    assert all(r["activated"] for r in creps), creps
    cons = {}
    max_diff = 0.0
    for k, _ in ORDER:
        b = rtc.batch(str(RUN / f"eval_{k}.jsonl"))
        d = max(abs(a - c) for x, y in zip(b["results"], fresh_probs(rt2, k)) for a, c in zip(x["probs"], y))
        max_diff = max(max_diff, d)
        cons[k] = {"accuracy": b["summary"]["accuracy"], "latency_us_p50": b["summary"]["latency_us_p50"],
                   "peak_rss_kb": b["summary"]["peak_rss_kb"], "max_prob_diff_vs_full": d}
    full = {k: {"accuracy": v["accuracy"], **{m: rt2.batch(str(RUN / f"eval_{k}.jsonl"))["summary"][m] for m in ("latency_us_p50", "peak_rss_kb")}} for k, v in fresh.items()}
    report["constrained"] = {"profile": "PC_CONSTRAINED (512MB RAM budget, 8MB model limit, 4 loaded modules; process: 1 CPU, 1GiB AS limit)",
                             "per_capability": cons, "full": full, "max_prob_diff_vs_full": max_diff,
                             "note": "simulation only; physical Android validation pending"}
    print("constrained max prob diff vs full:", max_diff)

    report["baselines"] = baselines.run()
    report["teacher_dependency"] = {"stages": teacher_log, "teacher_calls_total": teacher.calls,
                                    "related_encounters_without_teacher": sum(1 for s in stages if not s["related_encounter"]["teacher_called"]
                                                                              and s["related_encounter"]["result"] == "ANSWER")}
    out = ROOT / "benchmarks" / "reports" / "phase3.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print("wrote", out)


def fresh_probs(cli, task):
    b = cli.batch(str(RUN / f"eval_{task}.jsonl"))
    return [x["probs"] for x in b["results"]]


if __name__ == "__main__":
    main()
