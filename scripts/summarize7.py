"""Render benchmarks/reports/phase7.json (+ cross-phase baseline comparison) as markdown. Numbers come only from the JSON reports."""
import json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
rep = lambda n: json.loads((ROOT / f"benchmarks/reports/{n}.json").read_text())
r, p3, p4, p5, p6 = rep("phase7"), rep("phase3"), rep("phase4"), rep("phase5"), rep("phase6")
f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"


def wilson(p, n, z=1.96):
    if not n:
        return "n/a"
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"[{max(0, c - h):.2f}, {min(1, c + h):.2f}]"


c = r["consolidation"]
L = [f"# Phase 7 benchmark summary ({r['date']}, {r['platform']})", "", "## Consolidation of a deliberately bloated registry", "",
     f"Before: {c['before']}", f"After:  {c['after']}", "", "Reductions come from: merging a functional duplicate, archiving never-used capabilities (reversible), and distilling modules that stayed within 1 point of the original.", "",
     "### Functional-agreement scan (same input dim and label set only)", "", "| pair | agreement (in-distribution) | agreement (uniform box) | verdict |", "|---|---|---|---|"]
for x in c["pairwise_functional_agreement"]:
    L.append(f"| {x['a']} / {x['b']} | {f(x['agreement_in_distribution'])} | {f(x['agreement_uniform_box'])} | {x['verdict']} |")
m = c["merge_duplicate"]
L += ["", f"Merge `{m['duplicate']}` into `{m['keeper']}`: merged={m['merged']}, steps={[s['step'] for s in m['steps']]}, restore works={m['restore_works']}. Archive guard (capability with a dependent): refused={c['archive_guard']['refused']}. Archived as unused: {[a['capability_id'] for a in c['archive_unused']['archived']]}.", "",
      "### Distillation vs structured pruning (16 hidden units each; accuracy on the 3 held-out eval sets)", "", "| capability | params before | after | held-out acc before | distilled | pruned (dry run) | installed? | bytes before -> after | latency us p50 before -> after | rollback check |", "|---|---|---|---|---|---|---|---|---|---|"]
for e in c["compaction"]:
    d = e["distillation"]; rb = e["rollback_check"]
    L.append(f"| {e['capability']} | {e['before']['params']} | {d['params_after']} | {f(d['teacher_heldout'])} | {f(d['student_heldout'])} | {f(e['pruning_dry_run']['student_heldout'])} | {d['installed']}{'' if d['installed'] else ' (' + str(d['reason']) + ')'} | {e['before']['bytes']} -> {d.get('bytes_after') or e['before']['bytes']} | {e['before']['latency_us_p50']} -> {e['after']['latency_us_p50']} | {rb} |")
L += ["", "### Route optimisation (common-subexpression + dead-node elimination)", "", "| plan | cap calls before -> after | nodes before -> after | outputs identical | latency us p50 before -> after |", "|---|---|---|---|---|"]
for o in c["route_optimisation"]:
    L.append(f"| {o['plan']} | {o['cap_calls_before']} -> {o['cap_calls_after']} | {o['nodes_before']} -> {o['nodes_after']} | {o['outputs_identical']} | {o['latency_us_p50_before']} -> {o['latency_us_p50_after']} |")
rt = r["routing"]
L += ["", "## Learned router vs deterministic keyword router", "", f"Evaluation intents are hand-written and disjoint from training templates (overlap check: {rt['eval_train_overlap']}). **Small samples**: 95% Wilson intervals in brackets.", "",
      "| router | in-scope n | correct & KNOWN | routed correctly (any status) | needs help | OOS n | OOS fabrication (KNOWN) | adversarial n | adversarial fabrication (KNOWN) | adversarial rejected |", "|---|---|---|---|---|---|---|---|---|---|"]
def row(name, e):
    i, o, a = e["in_scope"], e["out_of_scope"], e["adversarial"]
    return f"| {name} | {i['n']} | {f(i['correct_known'], 2)} {wilson(i['correct_known'], i['n'])} | {f(i['routed_correct'], 2)} {wilson(i['routed_correct'], i['n'])} | {f(i['needs_help'], 2)} | {o['n']} | {f(o['fabrication_known'], 2)} {wilson(o['fabrication_known'], o['n'])} | {a['n']} | {f(a['fabrication_known'], 2)} {wilson(a['fabrication_known'], a['n'])} | {f(a['rejected'], 2)} |"
L.append(row("keyword", rt["keyword_router"]))
for k, v in rt["learned_router"].items():
    L.append(row(f"learned ({k} negatives)", v["eval"]))
tr = {k: v["training"] for k, v in rt["learned_router"].items()}
L += ["", f"Learned router: params {tr['hard']['params']}, {tr['hard']['bytes']} bytes, training {tr['hard']['total_seconds']:.1f}s. Per-task wall time over {rt['speed']['keyword']['cases']} tasks: keyword {rt['speed']['keyword']['us_per_task_wall']:.1f} us, learned {rt['speed']['learned']['us_per_task_wall']:.1f} us; peak RSS {rt['speed']['keyword']['peak_rss_kb']} vs {rt['speed']['learned']['peak_rss_kb']} KB.", "",
      "### A new capability arrives after the router was trained (4 paraphrased majority-vote intents)", "", "| router | correct & KNOWN | needs help |", "|---|---|---|"]
nc = rt["new_capability"]
for k in ("keyword_immediately", "learned_stale", "learned_after_retrain"):
    L.append(f"| {k} | {f(nc[k]['correct_known'], 2)} | {f(nc[k]['needs_help'], 2)} |")
L += ["", f"Learned-router retrain: {nc['retrain_seconds']:.1f}s per new capability; keyword router: 0 (reads package metadata).", ""]
# cross-phase baseline comparison
L += ["## Cross-phase baseline comparison (all numbers from the earlier reports)", "", "### Single capabilities (Phase 3)", "", "| capability | modular (3 separate modules) | one shared MLP trained jointly | one shared MLP fine-tuned sequentially, final |", "|---|---|---|---|"]
last = p3["stages"][-1]["accuracy_all_known"]; b = p3["baselines"]
for k in last:
    L.append(f"| {k} | {f(last[k]['accuracy'])} | {f(b['joint'][k])} | {f(b['sequential'][-1]['acc'][k])} |")
L += ["", f"Parameters: modular {sum(x['params'] for x in p3['registry'])} total vs shared MLP {b['params']}.", "", "### Composite tasks (Phase 4; end-to-end MLP trained on 3,000 composite examples)", "", "| composite | modular (hand-written plan) | end-to-end MLP (mean of 3 seeds) |", "|---|---|---|"]
for k, v in p4["composites"].items():
    best = max(x["accuracy"] for x in v["variants"].values())
    L.append(f"| {k} | {f(best)} | {f(v['baseline_end_to_end_mlp']['accuracy_mean'])} |")
lat = p4["latency_modular_vs_monolithic"]
L += ["", "Latency through the same ONNX backend: " + "; ".join(f"{k}: modular {v['modular_plan_total_us_p50']} us vs monolithic {v['monolithic_inference_us_p50']} us" for k, v in lat.items()) + ".", "",
      "### Teacher dependency", "", f"Phase 3 stream: 3 teacher calls for 3 first encounters, 0 for 3 related re-encounters. Phase 5 stream: related re-encounters answered locally {p5['escalation']['related_encounters_local']}/{p5['escalation']['related_encounters_total']}. "
      f"Phase 6 paraphrase test: {p6['paraphrase_teacher_dependency']['without_router_update']['teacher_calls_after_learning']} teacher calls without router_update vs {p6['paraphrase_teacher_dependency']['with_router_update']['teacher_calls_after_learning']} with it (36 tasks).", "",
      "### Not available here", "", "- Small quantized LLM baseline: **PENDING** (no model weights and no network egress in this environment). No substitute was used.", "- Energy/battery/CPU-GPU utilisation: not measured.", ""]
(ROOT / "benchmarks/reports/phase7.md").write_text("\n".join(L)); print("ok")
