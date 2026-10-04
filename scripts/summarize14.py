"""Render benchmarks/reports/phase14.json as markdown."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
r = json.loads((ROOT / "benchmarks/reports/phase14.json").read_text())
L = [f"# Phase 14 benchmark summary ({r['date']}, {r['platform']})", "", "**Caveats:** " + "; ".join(r["caveats"]) + ".", "",
     "## Arms on the same 723-arrival stream (36 capabilities, 3 rule drifts, unsupported requests)", "",
     "| arm | teacher calls | silent wrong answers (rate) | unserved | learned at first request | final acc mean / min | modules | drift repairs (delay) | spurious repairs |", "|---|---|---|---|---|---|---|---|---|"]
for n, a in r["arms"].items():
    rep = ", ".join(f"{c}:{x[0]['delay']}" for c, d in a["drift"].items() for x in [d["repairs"]] if x) or "none"
    L.append(f"| {n} | {a['teacher_calls']} | {a['silent_wrong_answers']} ({a['silent_wrong_rate']:.3f}) | {a['unserved_capability_arrivals']} | {a['new_capabilities_learned_at_first_request']}/{a['new_capabilities_requested']} | {a['final_accuracy_mean']:.3f} / {a['final_accuracy_min']:.2f} | {a['modules_final']['active_modules']} | {rep} | {len(a['spurious_repairs'])} |")
b = r["arms"]["base"]
L += ["", f"Always asking the teacher: {r['teacher_always']['teacher_calls']} calls, {r['teacher_always']['teacher_bytes_in']} request bytes (base: {b['teacher_calls']} calls, {b['teacher_bytes_in']} bytes).", "",
      "Teacher calls per 100 arrivals by window (base): " + ", ".join(map(str, b["teacher_calls_per_100_arrivals_by_window"])) + ".", "",
      "Teacher calls by action (base): " + json.dumps(b["teacher_calls_by_action"]) + ".", "",
      "## Single-network baselines (oracle routing, repaired when the modular system repaired)", "", "| baseline | final acc mean / min | backward transfer | params | train s |", "|---|---|---|---|---|"]
for m, x in r["baselines"].items(): L.append(f"| {m} | {x['final_accuracy_mean']:.3f} / {x['final_accuracy_min']:.2f} | {x['backward_transfer_stable_capabilities']:+.3f} | {x['params']} | {x['train_seconds']} |")
L.append(f"| modular (base) | {b['final_accuracy_mean']:.3f} / {b['final_accuracy_min']:.2f} | {b['backward_transfer_stable_capabilities']:+.4f} | {b['modules_final']['params']} (36 modules) | {b['build_seconds']} |")
L += ["", "## Seeds (base arm)", "", "| seed | calls | wrong (rate) | acc mean / min | learned at first request | drift delays |", "|---|---|---|---|---|---|"]
for s, x in sorted(r["seeds"].items()): L.append(f"| {s} | {x['teacher_calls']} | {x['silent_wrong_answers']} ({x['silent_wrong_rate']:.3f}) | {x['final_accuracy_mean']:.3f} / {x['final_accuracy_min']:.2f} | {x['new_capabilities_learned_at_first_request']}/36 | {x['drift_delays']} |")
p = b["post"]; ro = p["routing_at_scale"]
L += ["", "## Routing at 36 modules (unseen wordings)", "", f"Learned router validation accuracy {ro['learned_router']['val_accuracy']:.2f}.", "", "| router | set | correct KNOWN | wrong KNOWN | NEEDS_HELP | uncertain/other |", "|---|---|---|---|---|---|"]
for name in ("keyword_with_aliases_learned_in_stream", "learned"):
    for st, d in ro[name].items(): L.append(f"| {name} | {st} | {d['correct_known']:.2f} | {d['wrong_known']:.2f} | {d['needs_help']:.2f} | {d['other']:.2f} |")
c = p["consolidation"]
L += ["", "## Consolidation", "", f"Modules {c['before']['active_modules']} -> {c['after_merge']['active_modules']} after merging functional duplicates (params {c['before']['params']} -> {c['after_merge']['params']}); mean accuracy {c['accuracy_before']:.3f} -> {c['accuracy_after_merge']:.3f}.",
      f"Distillation to 16x16 under a 0.01 gate: {c['compaction']['installed']} installed, {c['compaction']['rejected']} rejected; params {c['params_before_compaction']} -> {c['after_compaction']['params']}; held-out {c['compaction']['teacher_heldout_mean']:.3f} -> {c['compaction']['student_heldout_mean']:.3f}; world accuracy {c['accuracy_after_merge']:.3f} -> {c['accuracy_after_compaction']:.3f}.", "",
      "| twin pair | agreement | verdict | rules identical in the world | merged |", "|---|---|---|---|---|"]
for t in c["twin_pairs"]: L.append(f"| {t['keeper']} / {t['duplicate']} | {t['agreement']} | {t['verdict']} | {t['rules_identical_in_world']} | {t.get('merge', {}).get('merged')} |")
L += ["", "Archive-by-usage policy: " + json.dumps(c["archive_policy"]), "", "## Core latency vs module count (base)", "", "| arrival | modules | p50 us | p95 us |", "|---|---|---|---|"]
for x in b["latency_by_checkpoint"]: L.append(f"| {x['t']} | {x['modules']} | {x['latency_us_p50']} | {x['latency_us_p95']} |")
(ROOT / "benchmarks/reports/phase14.md").write_text("\n".join(L)); print("ok")
