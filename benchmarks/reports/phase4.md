# Phase 4 benchmark summary (2026-10-04, Linux-6.18.44-fc-v64-x86_64-with-glibc2.39)

Module held-out accuracy: compare_numbers: 1.000, point_region: 0.992, argmax_position: 0.988

| composite | plan variant | modular acc | end-to-end MLP acc (3 seeds, mean) | MLP params | cap calls/task | node exec p50 | total us p50 | modules us | overhead us |
|---|---|---|---|---|---|---|---|---|---|
| sort4 | network/intent | 1.000 | 1.000 | 6040 | 5 | 16 | 88 | 37 | 50 |
| sort4 | network/id | 1.000 | 1.000 | 6040 | 5 | 16 | 76 | 43 | 32 |
| sort4 | bubble_loop/intent | 1.000 | 1.000 | 6040 | 9 | 29 | 165 | 70 | 93 |
| sort4 | bubble_loop/id | 1.000 | 1.000 | 6040 | 9 | 29 | 112 | 63 | 49 |
| count_inside | main/intent | 0.957 | 0.569 | 5061 | 4 | 5 | 78 | 34 | 43 |
| count_inside | main/id | 0.957 | 0.569 | 5061 | 4 | 5 | 52 | 30 | 22 |
| mixed_C_select_B_A | main/intent | 0.987 | 0.833 | 5126 | 3 | 6 | 66 | 30 | 35 |
| mixed_C_select_B_A | main/id | 0.987 | 0.833 | 5126 | 3 | 6 | 52 | 32 | 20 |
| branch | main/intent | 0.998 | 0.980 | 5516 | 1-2 | 3 | 35 | 16 | 19 |
| branch | main/id | 0.998 | 0.980 | 5516 | 1-2 | 3 | 27 | 16 | 11 |
| path_ABAC | main/intent | 0.993 | 0.862 | 9544 | 4 | 4 | 75 | 34 | 42 |
| path_ABAC | main/id | 0.993 | 0.862 | 9544 | 4 | 4 | 50 | 30 | 20 |

Intent-routed vs id-routed variants give identical accuracy; the difference is overhead only.

Branch exclusivity (only the taken side runs): True; `then` taken in 48.3% of cases.
A->B->A->C path executed in order for every case: True.

## Modular vs monolithic latency (same ONNX backend)

| task | monolithic us p50 | modular plan total us p50 | ratio |
|---|---|---|---|
| count_inside | 9 | 78 | 8.7x |
| mixed_C_select_B_A | 7 | 66 | 9.4x |

## Error accumulation check

Predicted accuracy if module errors were independent: count_inside 0.968, mixed 0.980, path_ABAC 0.980. Compare with the measured values above.

Loop safety: {'repeat_1e6_rejected_at_validation': True, 'nested_1000x1000_aborted_by_step_budget': True}
