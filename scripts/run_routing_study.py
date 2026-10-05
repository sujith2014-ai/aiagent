"""Routing generalisation study (Phase 14 follow-up). 36 capabilities learned from the simulated teacher, then routers compared on wordings none of them was trained on:
canonical intents, easy paraphrases (the noun is kept), hard paraphrases (noun and kind both replaced by synonyms), out-of-scope requests and adversarial word overlap.
Teacher-supplied synonym coverage p in {0, 0.5, 1.0} (see BankTeacher). Routers: keyword (current default), idf with unmatched-word weight w, learned."""
from __future__ import annotations
import collections, json, random, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.cli import ROOT
from scripts.phase14_lab import World, make_stream, ModularRun
from scripts.run_phase14 import routing_rows_and_learned

RUN = ROOT / "runs" / "routing_study"
ROUTERS = [("keyword", None), ("idf:0.25", "idf:0.25"), ("idf:0.5", "idf:0.5"), ("idf:1.0", "idf:1.0"), ("learned", "learned")]
ADV = ["check the weather alarm", "check the stock zone", "run the lunch menu test", "check the alarm clock on my phone", "check the side of the road", "run the rank test on my chess score",
       "check the grade of this essay", "check the peak season prices", "run the vote count on the ballot", "check the order status of my package"]


def main():
    if RUN.exists(): shutil.rmtree(RUN)
    RUN.mkdir(parents=True)
    out = {"benchmark_format": "bench/1", "experiment": "routing-generalisation-study", "date": time.strftime("%Y-%m-%d"), "coverage": {}}
    for p in (0.0, 0.5, 1.0):
        w = World(0); stream = [e for e in make_stream(w, seed=0) if e["type"] == "new"]
        for i, e in enumerate(stream): e["t"] = i
        run = ModularRun(RUN / f"p{p}", w, monitor=False, synonym_coverage=p, accept_synonyms=p > 0).run(stream, 10_000)
        rows, learned_rep = routing_rows_and_learned(run, w, ADV)
        res = {"modules": len(run.cli.list()), "learned_router_val_accuracy": learned_rep["val_accuracy"], "routers": {}}
        for name, router in ROUTERS:
            f = run.tmp / f"r_{name}.jsonl"; f.write_text("\n".join(json.dumps({"intent": r["intent"], "input": r["input"]}) for r in rows) + "\n")
            got = run.cli._run_detect("full", "batch", "--cases", str(f), router=router)["results"]
            m = collections.defaultdict(lambda: {"n": 0, "correct_known": 0, "wrong_known": 0, "not_known": 0})
            for r, x in zip(rows, got):
                d = m[r["set"]]; d["n"] += 1
                known = (not x.get("needs_help")) and x.get("status") == "KNOWN"
                if not known: d["not_known"] += 1
                elif r["gold"] == "UNKNOWN" or x.get("capability") != r["gold"]: d["wrong_known"] += 1
                else: d["correct_known"] += 1
            res["routers"][name] = {k: {kk: round(vv / v["n"], 3) if kk != "n" else vv for kk, vv in v.items()} for k, v in m.items()}
        out["coverage"][str(p)] = res; print("p", p, "done", flush=True)
    (ROOT / "benchmarks/reports/routing_study.json").write_text(json.dumps(out, indent=1)); print("ok")


if __name__ == "__main__":
    main()
