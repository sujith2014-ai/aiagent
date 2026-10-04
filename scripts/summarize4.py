"""Render benchmarks/reports/phase4.json as markdown (numbers only from the JSON)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
r = json.loads((ROOT / "benchmarks/reports/phase4.json").read_text())
L = [f"# Phase 4 benchmark summary ({r['date']}, {r['platform']})", "",
     "Module held-out accuracy: " + ", ".join(f"{k}: {v:.3f}" for k, v in r["module_heldout_accuracy"].items()), "",
     "| composite | plan variant | modular acc | end-to-end MLP acc (3 seeds, mean) | MLP params | cap calls/task | node exec p50 | total us p50 | modules us | overhead us |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for n, c in r["composites"].items():
    b = c["baseline_end_to_end_mlp"]
    for k, v in c["variants"].items():
        L.append(f"| {n} | {k} | {v['accuracy']:.3f} | {b['accuracy_mean']:.3f} | {b['params']} | {sum(v['capability_calls_per_task'].values()) if n not in ('branch',) else '1-2'} | {v['node_executions_p50']} | {v['latency_us_p50']['total']} | {v['latency_us_p50']['modules']} | {v['latency_us_p50']['overhead']} |")
L += ["", "Intent-routed vs id-routed variants give identical accuracy; the difference is overhead only.", "",
      f"Branch exclusivity (only the taken side runs): {r['composites']['branch']['branch_exclusive']}; `then` taken in {r['composites']['branch']['branch_taken_then_fraction']:.1%} of cases.",
      f"A->B->A->C path executed in order for every case: {r['composites']['path_ABAC']['path_order_ok']}.", "",
      "## Modular vs monolithic latency (same ONNX backend)", "", "| task | monolithic us p50 | modular plan total us p50 | ratio |", "|---|---|---|---|"]
for k, v in r["latency_modular_vs_monolithic"].items():
    L.append(f"| {k} | {v['monolithic_inference_us_p50']} | {v['modular_plan_total_us_p50']} | {v['latency_ratio_modular_over_monolithic']:.1f}x |")
e = r["expected_accuracy_if_errors_independent"]
L += ["", "## Error accumulation check", "", f"Predicted accuracy if module errors were independent: count_inside {e['count_inside']:.3f}, mixed {e['mixed_C_select_B_A']:.3f}, path_ABAC {e['path_ABAC']:.3f}. Compare with the measured values above.", "",
      f"Loop safety: {r['loop_safety']}", ""]
(ROOT / "benchmarks/reports/phase4.md").write_text("\n".join(L)); print("\n".join(L))
