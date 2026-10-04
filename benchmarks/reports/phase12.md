# Phase 12 benchmark summary (2026-10-04, Linux-6.18.44-fc-v64-x86_64-with-glibc2.39)

On-device runs: taskset -c 0 + RLIMIT_AS 256MB for on-device runs. **Not measured:** battery, thermal throttling, NPU/GPU, real phone CPU (this is a PC core pinned to one CPU).

## Training from scratch: device (Rust trainer) vs server (PyTorch build service), 5 random 70/30 splits per task

| task | train / eval rows | device acc (mean±sd) | server acc (mean±sd) | device train s | device wall s* | device peak RSS MB | device pkg bytes | server train+sign s | server pkg bytes |
|---|---|---|---|---|---|---|---|---|---|
| iris_species | 105 / 45 | 0.907±0.090 | 0.880±0.061 | 0.048 | 0.056 | 6.8 | 7459 | 0.39 | 7378 |
| wine_cultivar | 124 / 54 | 0.933±0.034 | 0.933±0.040 | 0.087 | 0.096 | 6.8 | 9973 | 0.12 | 9287 |
| digit_recognition | 1257 / 540 | 0.958±0.004 | 0.955±0.012 | 0.738 | 0.770 | 8.3 | 26470 | 0.71 | 26981 |
| tumor_malignancy | 398 / 171 | 0.958±0.008 | 0.959±0.014 | 0.178 | 0.193 | 7.1 | 28106 | 0.22 | 18446 |

*wall includes spawning the CLI process, building, signing and importing the package.
Server process peak RSS (whole harness, includes PyTorch and scikit-learn): 809.6 MB.

## Few-shot adaptation to an input drift (two highest-variance features read 1.5 sd high), 5 draws

`frozen` = the unchanged model. `head`/`full` = on-device adaptation with the acceptance gates disabled (raw effect). `new_small_module` = train a fresh module on only the new samples. `server_retrain` = PyTorch on old + new data (the server holds the data; the device does not).

'gate acceptance' = share of draws in which the default gates (better on new data, old-domain accuracy not down by more than 0.05) would have accepted the update.

### iris_species (drifted feature columns [0, 2])

| n adaptation samples | method | new-domain acc | old-domain acc | train s | gate acceptance |
|---|---|---|---|---|---|
| 10 | frozen | 0.626±0.137 | 0.947 | 0.000 | 0.0 |
| 10 | head | 0.858±0.068 | 0.847 | 0.002 | 0.4 |
| 10 | full | 0.853±0.114 | 0.847 | 0.003 | 0.4 |
| 25 | frozen | 0.626±0.137 | 0.947 | 0.000 | 0.0 |
| 25 | head | 0.916±0.056 | 0.832 | 0.004 | 0.2 |
| 25 | full | 0.900±0.061 | 0.826 | 0.007 | 0.2 |
| 25 | server_retrain | 0.816±0.110 | 0.853 | 0.129 | 0.2 |

New small module: n=10: 5/5 refused (Error: need at least 30 distinct examples, got 10), n=25: 5/5 refused (Error: need at least 30 distinct examples, got 25). Server retrain: n=10: 5/5 failed (invalid examples: validation tests missing/mismatched (need >= 20)).

### wine_cultivar (drifted feature columns [4, 12])

| n adaptation samples | method | new-domain acc | old-domain acc | train s | gate acceptance |
|---|---|---|---|---|---|
| 10 | frozen | 0.858±0.057 | 0.951 | 0.000 | 0.0 |
| 10 | head | 0.871±0.051 | 0.969 | 0.001 | 0.4 |
| 10 | full | 0.911±0.024 | 0.964 | 0.004 | 0.8 |
| 10 | server_retrain | 0.956±0.024 | 0.951 | 0.120 | 1.0 |
| 25 | frozen | 0.858±0.057 | 0.951 | 0.000 | 0.0 |
| 25 | head | 0.938±0.017 | 0.956 | 0.004 | 0.6 |
| 25 | full | 0.960±0.036 | 0.942 | 0.009 | 0.6 |
| 25 | server_retrain | 0.969±0.018 | 0.973 | 0.126 | 1.0 |

New small module: n=10: 5/5 refused (Error: need at least 30 distinct examples, got 10), n=25: 5/5 refused (Error: need at least 30 distinct examples, got 25).

### digit_recognition (drifted feature columns [43, 42])

| n adaptation samples | method | new-domain acc | old-domain acc | train s | gate acceptance |
|---|---|---|---|---|---|
| 10 | frozen | 0.907±0.011 | 0.941 | 0.000 | 0.0 |
| 10 | head | 0.892±0.029 | 0.913 | 0.003 | 0.2 |
| 10 | full | 0.890±0.030 | 0.913 | 0.006 | 0.4 |
| 10 | server_retrain | 0.922±0.019 | 0.949 | 0.723 | 0.8 |
| 25 | frozen | 0.907±0.011 | 0.941 | 0.000 | 0.0 |
| 25 | head | 0.914±0.011 | 0.935 | 0.003 | 0.4 |
| 25 | full | 0.916±0.007 | 0.930 | 0.009 | 1.0 |
| 25 | server_retrain | 0.921±0.006 | 0.952 | 0.658 | 1.0 |

New small module: n=10: 5/5 refused (Error: need at least 30 distinct examples, got 10), n=25: 5/5 refused (Error: need at least 30 distinct examples, got 25).

### tumor_malignancy (drifted feature columns [3, 23])

| n adaptation samples | method | new-domain acc | old-domain acc | train s | gate acceptance |
|---|---|---|---|---|---|
| 10 | frozen | 0.895±0.023 | 0.940 | 0.000 | 0.0 |
| 10 | head | 0.913±0.035 | 0.936 | 0.001 | 0.6 |
| 10 | full | 0.916±0.027 | 0.931 | 0.004 | 0.6 |
| 10 | server_retrain | 0.931±0.020 | 0.959 | 0.223 | 1.0 |
| 25 | frozen | 0.895±0.023 | 0.940 | 0.000 | 0.0 |
| 25 | head | 0.943±0.014 | 0.940 | 0.006 | 1.0 |
| 25 | full | 0.944±0.015 | 0.945 | 0.013 | 1.0 |
| 25 | server_retrain | 0.944±0.035 | 0.952 | 0.207 | 1.0 |

New small module: n=10: 5/5 refused (Error: need at least 30 distinct examples, got 10), n=25: 5/5 refused (Error: need at least 30 distinct examples, got 25).
