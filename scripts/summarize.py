"""Render benchmarks/reports/phase3.json as a human-readable markdown summary (numbers only come from the JSON)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
r = json.loads((ROOT / "benchmarks/reports/phase3.json").read_text())
L = [f"# Phase 3 benchmark summary ({r['date']}, {r['platform']})", ""]
L += ["## Sequential learning (modular system)", "", "| stage | learned | acc of each known capability (unseen eval set) | bundled regression | teacher called for related task |", "|---|---|---|---|---|"]
for i, s in enumerate(r["stages"]):
    acc = ", ".join(f"{k}: {v['accuracy']:.3f}" for k, v in s["accuracy_all_known"].items())
    reg = ", ".join(f"{k}: {v:.3f}" for k, v in s["bundled_regression"].items())
    L.append(f"| {i+1} | {s['learned']} | {acc} | {reg} | {s['related_encounter']['teacher_called']} |")
L += ["", f"Max forgetting drop (modular): **{r['max_forgetting_drop']:.4f}**", ""]
b = r["baselines"]
L += [f"## Conventional tiny-network baseline (shared MLP, hidden {b['hidden']}, {b['params']} params)", "",
      "Joint training (all tasks at once): " + ", ".join(f"{k}: {v:.3f}" for k, v in b["joint"].items()), "",
      "Naive sequential fine-tuning of one shared network:", "", "| after learning | accuracies |", "|---|---|"]
for s in b["sequential"]:
    L.append(f"| {s['after']} | " + ", ".join(f"{k}: {v:.3f}" for k, v in s["acc"].items()) + " |")
tot_p = sum(x["params"] for x in r["registry"]); tot_b = sum(x["model_bytes"] for x in r["registry"])
L += ["", f"Modular system total: {tot_p} params, {tot_b} bytes of ONNX across {len(r['registry'])} modules (baseline shared MLP: {b['params']} params).", ""]
cr = r["clean_restart"]
L += ["## Clean restart (export -> destroy runtime state -> fresh install -> import)", "",
      f"Fresh runtime imported {cr['fresh_runtime_imports']} packages; outputs bit-identical to pre-restart: **{cr['all_identical']}**", ""]
c = r["constrained"]
L += [f"## Constrained device simulation", "", c["profile"], "", "| capability | acc (constrained) | acc (full) | p50 latency us (constr./full) | peak RSS KB (constr./full) |", "|---|---|---|---|---|"]
for k, v in c["per_capability"].items():
    f = c["full"][k]
    L.append(f"| {k} | {v['accuracy']:.3f} | {f['accuracy']:.3f} | {v['latency_us_p50']} / {f['latency_us_p50']} | {v['peak_rss_kb']} / {f['peak_rss_kb']} |")
L += ["", f"Max probability difference constrained vs full: {c['max_prob_diff_vs_full']}", "", f"_{c['note']}_", ""]
t = r["teacher_dependency"]
L += ["## Teacher dependency", "", f"Teacher calls total: {t['teacher_calls_total']} (one per new capability). Related re-encounters answered locally without the teacher: {t['related_encounters_without_teacher']}/{len(r['stages'])}.", ""]
(ROOT / "benchmarks/reports/phase3.md").write_text("\n".join(L))
print("\n".join(L))
