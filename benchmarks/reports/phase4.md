# Phase 4 benchmark summary (2026-10-04, Linux-6.18.44-fc-v64-x86_64-with-glibc2.39)

Module held-out accuracy: compare_numbers: 1.000, point_region: 0.992, argmax_position: 0.988

| composite | plan variant | modular acc | end-to-end MLP acc (3 seeds, mean) | MLP params | cap calls/task | node exec p50 | total us p50 | modules us | overhead us |
|---|---|---|---|---|---|---|---|---|---|
| sort4 | network/intent | 1.000 | 1.000 | 6040 | 5 | 16 | 56 | 7 | 49 |
| sort4 | network/id | 1.000 | 1.000 | 6040 | 5 | 16 | 32 | 8 | 24 |
| sort4 | bubble_loop/intent | 1.000 | 1.000 | 6040 | 9 | 29 | 81 | 9 | 70 |
| sort4 | bubble_loop/id | 1.000 | 1.000 | 6040 | 9 | 29 | 37 | 9 | 27 |
| count_inside | main/intent | 0.957 | 0.569 | 5061 | 4 | 5 | 28 | 4 | 24 |
| count_inside | main/id | 0.957 | 0.569 | 5061 | 4 | 5 | 13 | 4 | 8 |
| mixed_C_select_B_A | main/intent | 0.987 | 0.833 | 5126 | 3 | 6 | 23 | 3 | 20 |
| mixed_C_select_B_A | main/id | 0.987 | 0.833 | 5126 | 3 | 6 | 11 | 3 | 8 |
| branch | main/intent | 0.998 | 0.980 | 5516 | 1-2 | 3 | 14 | 2 | 11 |
| branch | main/id | 0.998 | 0.980 | 5516 | 1-2 | 3 | 8 | 2 | 5 |
| path_ABAC | main/intent | 0.993 | 0.862 | 9544 | 4 | 4 | 39 | 7 | 32 |
| path_ABAC | main/id | 0.993 | 0.862 | 9544 | 4 | 4 | 14 | 4 | 9 |

Intent-routed vs id-routed variants give identical accuracy; the difference is overhead only.

Branch exclusivity (only the taken side runs): True; `then` taken in 48.3% of cases.
A->B->A->C path executed in order for every case: True.

## Modular vs monolithic latency (same ONNX backend)

| task | monolithic us p50 | modular plan total us p50 | ratio |
|---|---|---|---|
| count_inside | 7 | 28 | 4.0x |
| mixed_C_select_B_A | 4 | 23 | 5.8x |

## Error accumulation check

Predicted accuracy if module errors were independent: count_inside 0.968, mixed 0.980, path_ABAC 0.980. Compare with the measured values above.

Loop safety: {'repeat_1e6_rejected_at_validation': True, 'nested_1000x1000_aborted_by_step_budget': True}
