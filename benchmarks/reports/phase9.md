# Phase 9 benchmark summary (2026-10-04, Linux-6.18.44-fc-v64-x86_64-with-glibc2.39)

Data: scikit-learn bundled copies of iris, wine, digits, breast cancer (duplicates removed, stratified 60/20/20 split, seed 0; baselines fit on the same 60% train split).

Taught by example through the desktop app: **no teacher, no rule**. Accuracy is measured through the Rust runtime on the held-out 20% (refused rows count as misses in `overall`; `answered` excludes them). Intervals are 95% Wilson; test sets are small (30-360 rows).

| task | n | gate | modular overall [95% CI] | coverage | answered acc | logistic reg. | kNN k=5 | sklearn MLP | majority | params (ours / LR / MLP) | model bytes | latency us p50 | ECE raw -> calibrated |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iris_species | 149 | 0.937 | 1.000 [0.886, 1] | 1.000 | 1.000 | 0.967 | 0.967 | 1.000 | 0.333 | 4675 / 15 / 1315 | 19528 | 11 | 0.037 -> 0.000 |
| wine_cultivar | 178 | 0.942 | 1.000 [0.904, 1] | 1.000 | 1.000 | 0.972 | 0.889 | 0.972 | 0.417 | 1603 / 42 / 1603 | 7309 | 12 | 0.037 -> 0.000 |
| digit_recognition | 1797 | 0.934 | 0.950 [0.922, 0.968] | 0.969 | 0.980 | 0.964 | 0.969 | 0.958 | 0.103 | 3466 / 650 / 3466 | 15173 | 9 | 0.010 -> 0.010 |
| tumor_malignancy | 569 | 0.950 | 0.965 [0.913, 0.986] | 0.991 | 0.973 | 0.982 | 0.939 | 0.974 | 0.623 | 2114 / 31 / 2081 | 9491 | 8 | 0.025 -> 0.028 |

Max forgetting drop across the 4 sequential stages: 0.0 (independent modules: zero by construction).

Routing by intent+input shape with all four installed (correct capability fraction): {'iris_species': 1.0, 'wine_cultivar': 1.0, 'digit_recognition': 1.0, 'tumor_malignancy': 0.975}.
Inputs shifted far out of range refused: {'iris_species': 1.0, 'wine_cultivar': 1.0, 'digit_recognition': 1.0, 'tumor_malignancy': 1.0}.

## Novelty rule on the real tasks (Rust runtime)

| task | strict: held-out coverage / +4sd shift refused | balanced: held-out coverage / +4sd shift refused |
|---|---|---|
| iris_species | 1.000 / 1.000 | 1.000 / 1.000 |
| wine_cultivar | 0.944 / 1.000 | 1.000 / 1.000 |
| digit_recognition | 0.986 / 1.000 | 0.969 / 1.000 |
| tumor_malignancy | 0.956 / 0.877 | 0.991 / 0.728 |

Mean over tasks: strict coverage 0.972, shift refused 0.969; balanced coverage 0.990, shift refused 0.932.

### Offline rule comparison (mean over the 4 datasets; benchmarks/reports/novelty_rules.json)

| rule | false refusal on held-out | gross | +4 sd | -4 sd | single feature +25 sd | in-range uniform |
|---|---|---|---|---|---|---|
| R0 any feature outside range+-10% (current) | 0.03 | 1.00 | 0.97 | 1.00 | 0.98 | 0.00 |
| R1 >=25% of features outside range+-10% | 0.00 | 1.00 | 0.83 | 1.00 | 0.25 | 0.00 |
| R2 any feature |z|>8 | 0.01 | 1.00 | 0.05 | 0.01 | 1.00 | 0.32 |
| R3 any feature |z|>12 | 0.01 | 1.00 | 0.01 | 0.00 | 1.00 | 0.25 |
| R4 >=25% outside range+-10% OR any |z|>12 | 0.01 | 1.00 | 0.83 | 1.00 | 1.00 | 0.25 |
| R5 >=10% outside range+-10% OR any |z|>12 | 0.01 | 1.00 | 0.89 | 1.00 | 1.00 | 0.25 |
| R6 >=2 features outside range+-10% OR any |z|>12 | 0.01 | 1.00 | 0.93 | 1.00 | 1.00 | 0.25 |
| R7 >=2 features outside range+-10% OR any |z|>8 | 0.01 | 1.00 | 0.93 | 1.00 | 1.00 | 0.31 |
| R8 >=2 outside range+-10% OR any |z|>12 (std floored at 5% of range) | 0.01 | 1.00 | 0.93 | 1.00 | 1.00 | 0.25 |
| R9 >=2 outside range+-10% OR any |z|>8 (std floored at 5% of range) | 0.01 | 1.00 | 0.93 | 1.00 | 1.00 | 0.31 |

## Clean restart and constrained device

Fresh runtime import ok: True; constrained profile import ok: True.

| task | identical outputs after restart | constrained max prob diff | same refusals | constrained peak RSS KB |
|---|---|---|---|---|
| iris_species | True | 0.0 | True | 13752 |
| wine_cultivar | True | 0.0 | True | 14024 |
| digit_recognition | True | 0.0 | True | 14728 |
| tumor_malignancy | True | 0.0 | True | 14404 |