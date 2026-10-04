# Phase 7 benchmark summary (2026-10-04, Linux-6.18.44-fc-v64-x86_64-with-glibc2.39)

## Consolidation of a deliberately bloated registry

Before: {'active_capabilities': 7, 'archived': 0, 'active_params': 8851, 'active_model_bytes': 39947}
After:  {'active_capabilities': 4, 'archived': 3, 'active_params': 3243, 'active_model_bytes': 15560}

Reductions come from: merging a functional duplicate, archiving never-used capabilities (reversible), and distilling modules that stayed within 1 point of the original.

### Functional-agreement scan (same input dim and label set only)

| pair | agreement (in-distribution) | agreement (uniform box) | verdict |
|---|---|---|---|
| compare_numbers / compare_values | 1.000 | 0.979 | duplicate |
| compare_numbers / compare_wrapper | 1.000 | 1.000 | duplicate |
| compare_values / compare_wrapper | 1.000 | 0.979 | duplicate |
| point_region / point_region__ext | 0.758 | 0.757 | overlapping_different_function |

Merge `compare_values` into `compare_numbers`: merged=True, steps=['keeper_passes_duplicate_tests', 'import_keeper_with_merged_keywords', 'routing_regression_after_archiving_duplicate'], restore works=True. Archive guard (capability with a dependent): refused=True. Archived as unused: ['compare_wrapper', 'majority_vote'].

### Distillation vs structured pruning (16 hidden units each; accuracy on the 3 held-out eval sets)

| capability | params before | after | held-out acc before | distilled | pruned (dry run) | installed? | bytes before -> after | latency us p50 before -> after | rollback check |
|---|---|---|---|---|---|---|---|---|---|
| compare_numbers | 1251 | 371 | 1.000 | 0.975 | 0.975 | False (accuracy drop 0.025 > 0.01) | 5653 -> 5653 | 8 -> 7 | {'skipped': 'nothing was installed'} |
| point_region | 1218 | 354 | 0.992 | 0.990 | 0.991 | True | 5521 -> 2061 | 7 -> 8 | {'active_version': '0.1.0', 'accuracy': 0.992, 'original_accuracy': 0.992} |
| argmax_position | 1348 | 420 | 0.988 | 0.982 | 0.984 | True | 6041 -> 2325 | 9 -> 9 | {'active_version': '0.1.0', 'accuracy': 0.987, 'original_accuracy': 0.987} |

### Route optimisation (common-subexpression + dead-node elimination)

| plan | cap calls before -> after | nodes before -> after | outputs identical | latency us p50 before -> after |
|---|---|---|---|---|
| count_inside | 8 -> 4 | 9 -> 5 | True | 86 -> 39 |
| sort4 | 10 -> 5 | 21 -> 16 | True | 113 -> 57 |

## Learned router vs deterministic keyword router

Evaluation intents are hand-written and disjoint from training templates (overlap check: []). **Small samples**: 95% Wilson intervals in brackets.

| router | in-scope n | correct & KNOWN | routed correctly (any status) | needs help | OOS n | OOS fabrication (KNOWN) | adversarial n | adversarial fabrication (KNOWN) | adversarial rejected |
|---|---|---|---|---|---|---|---|---|---|
| keyword | 27 | 0.44 [0.28, 0.63] | 0.67 [0.48, 0.81] | 0.33 | 24 | 0.04 [0.01, 0.20] | 15 | 0.20 [0.07, 0.45] | 0.13 |
| learned (basic negatives) | 27 | 0.89 [0.72, 0.96] | 0.93 [0.77, 0.98] | 0.07 | 24 | 0.12 [0.04, 0.31] | 15 | 0.60 [0.36, 0.80] | 0.20 |
| learned (hard negatives) | 27 | 0.81 [0.63, 0.92] | 0.89 [0.72, 0.96] | 0.11 | 24 | 0.08 [0.02, 0.26] | 15 | 0.27 [0.11, 0.52] | 0.60 |

Learned router: params 16708, 67274 bytes, training 1.6s. Per-task wall time over 10800 tasks: keyword 30.0 us, learned 43.2 us; peak RSS 26736 vs 27500 KB.

### A new capability arrives after the router was trained (4 paraphrased majority-vote intents)

| router | correct & KNOWN | needs help |
|---|---|---|
| keyword_immediately | 0.75 | 0.25 |
| learned_stale | 0.00 | 1.00 |
| learned_after_retrain | 0.50 | 0.50 |

Learned-router retrain: 2.1s per new capability; keyword router: 0 (reads package metadata).

## Cross-phase baseline comparison (all numbers from the earlier reports)

### Single capabilities (Phase 3)

| capability | modular (3 separate modules) | one shared MLP trained jointly | one shared MLP fine-tuned sequentially, final |
|---|---|---|---|
| argmax_position | 0.987 | 0.983 | 0.982 |
| compare_numbers | 1.000 | 1.000 | 0.038 |
| point_region | 0.992 | 0.992 | 0.605 |

Parameters: modular 3817 total vs shared MLP 2932.

### Composite tasks (Phase 4; end-to-end MLP trained on 3,000 composite examples)

| composite | modular (hand-written plan) | end-to-end MLP (mean of 3 seeds) |
|---|---|---|
| sort4 | 1.000 | 1.000 |
| count_inside | 0.957 | 0.569 |
| mixed_C_select_B_A | 0.987 | 0.833 |
| branch | 0.998 | 0.980 |
| path_ABAC | 0.993 | 0.862 |

Latency through the same ONNX backend: count_inside: modular 78 us vs monolithic 9 us; mixed_C_select_B_A: modular 66 us vs monolithic 7 us.

### Teacher dependency

Phase 3 stream: 3 teacher calls for 3 first encounters, 0 for 3 related re-encounters. Phase 5 stream: related re-encounters answered locally 10/12. Phase 6 paraphrase test: 9 teacher calls without router_update vs 3 with it (36 tasks).

### Not available here

- Small quantized LLM baseline: **PENDING** (no model weights and no network egress in this environment). No substitute was used.
- Energy/battery/CPU-GPU utilisation: not measured.
