# Phase 3 benchmark summary (2026-10-04, Linux-6.18.44-fc-v64-x86_64-with-glibc2.39)

## Sequential learning (modular system)

| stage | learned | acc of each known capability (unseen eval set) | bundled regression | teacher called for related task |
|---|---|---|---|---|
| 1 | compare_numbers | compare_numbers: 1.000 | compare_numbers: 1.000 | False |
| 2 | point_region | compare_numbers: 1.000, point_region: 0.992 | compare_numbers: 1.000, point_region: 1.000 | False |
| 3 | argmax_position | argmax_position: 0.987, compare_numbers: 1.000, point_region: 0.992 | argmax_position: 0.987, compare_numbers: 1.000, point_region: 1.000 | False |

Max forgetting drop (modular): **0.0000**

## Conventional tiny-network baseline (shared MLP, hidden [48, 48], 2932 params)

Joint training (all tasks at once): compare_numbers: 1.000, point_region: 0.992, argmax_position: 0.983

Naive sequential fine-tuning of one shared network:

| after learning | accuracies |
|---|---|
| compare_numbers | compare_numbers: 1.000 |
| point_region | compare_numbers: 0.038, point_region: 0.993 |
| argmax_position | compare_numbers: 0.038, point_region: 0.605, argmax_position: 0.982 |

Modular system total: 3817 params, 17215 bytes of ONNX across 3 modules (baseline shared MLP: 2932 params).

## Clean restart (export -> destroy runtime state -> fresh install -> import)

Fresh runtime imported 3 packages; outputs bit-identical to pre-restart: **True**

## Constrained device simulation

PC_CONSTRAINED (512MB RAM budget, 8MB model limit, 4 loaded modules; process: 1 CPU, 1GiB AS limit)

| capability | acc (constrained) | acc (full) | p50 latency us (constr./full) | peak RSS KB (constr./full) |
|---|---|---|---|---|
| compare_numbers | 1.000 | 1.000 | 24 / 30 | 13284 / 12836 |
| point_region | 0.992 | 0.992 | 27 / 21 | 13640 / 13828 |
| argmax_position | 0.987 | 0.987 | 21 / 24 | 14244 / 14436 |

Max probability difference constrained vs full: 0.0

_simulation only; physical Android validation pending_

## Teacher dependency

Teacher calls total: 3 (one per new capability). Related re-encounters answered locally without the teacher: 3/3.
