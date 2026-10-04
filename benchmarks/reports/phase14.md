# Phase 14 benchmark summary (2026-10-04, Linux-6.18.44-fc-v64-x86_64-with-glibc2.39)

**Caveats:** teacher is a simulator that knows the world (including drift); no live LLM; synthetic rule families, 36 capabilities, one world seed for the ablations; baselines get oracle routing and borrow the modular system's repair times; feedback is an oracle that labels a random 30% of served answers.

## Arms on the same 723-arrival stream (36 capabilities, 3 rule drifts, unsupported requests)

| arm | teacher calls | silent wrong answers (rate) | unserved | learned at first request | final acc mean / min | modules | drift repairs (delay) | spurious repairs |
|---|---|---|---|---|---|---|---|---|
| base | 185 | 41 (0.063) | 6 | 36/36 | 0.970 / 0.78 | 36 | none | 1 |
| serve_uncertain | 107 | 47 (0.073) | 10 | 8/36 | 0.970 / 0.78 | 27 | none | 0 |
| no_frame_filter | 154 | 72 (0.121) | 60 | 13/36 | 0.962 / 0.78 | 18 | none | 1 |
| cache_refusals | 147 | 41 (0.063) | 6 | 36/36 | 0.970 / 0.78 | 36 | none | 1 |
| no_router_update | 292 | 28 (0.043) | 6 | 36/36 | 0.970 / 0.78 | 36 | none | 0 |
| no_monitor | 184 | 41 (0.063) | 6 | 36/36 | 0.970 / 0.78 | 36 | none | 0 |
| feedback_0.1 | 184 | 41 (0.063) | 6 | 36/36 | 0.970 / 0.78 | 36 | none | 0 |
| feedback_1.0 | 207 | 23 (0.036) | 10 | 36/36 | 0.976 / 0.80 | 36 | boiler_alarm:139 | 2 |

Always asking the teacher: 720 calls, 456736 request bytes (base: 185 calls, 120185 bytes).

Teacher calls per 100 arrivals by window (base): 39.0, 25.0, 32.0, 23.0, 23.0, 21.0, 20.0, 5.0.

Teacher calls by action (base): {"new_capability_spec": 37, "reroute": 84, "request_tool": 25, "cannot_help": 19, "use_memory": 20}.

## Single-network baselines (oracle routing, repaired when the modular system repaired)

| baseline | final acc mean / min | backward transfer | params | train s |
|---|---|---|---|---|
| finetune | 0.520 / 0.14 | -0.041 | 22404 | 20.3 |
| replay | 0.950 / 0.75 | +0.002 | 22404 | 84.6 |
| joint | 0.960 / 0.78 | +0.000 | 22404 | 357.2 |
| modular (base) | 0.970 / 0.78 | -0.0009 | 48089 (36 modules) | 47.9 |

## Seeds (base arm)

| seed | calls | wrong (rate) | acc mean / min | learned at first request | drift delays |
|---|---|---|---|---|---|
| 0 | 185 | 41 (0.063) | 0.970 / 0.78 | 36/36 | {'boiler_alarm': [], 'gear_zone': [], 'mixer_zone': []} |
| 1 | 183 | 35 (0.054) | 0.972 / 0.73 | 36/36 | {'boiler_alarm': [], 'gear_zone': [], 'mixer_zone': []} |
| 2 | 185 | 46 (0.070) | 0.975 / 0.77 | 36/36 | {'boiler_alarm': [234], 'gear_zone': [], 'mixer_zone': []} |

## Routing at 36 modules (unseen wordings)

Learned router validation accuracy 0.69.

| router | set | correct KNOWN | wrong KNOWN | NEEDS_HELP | uncertain/other |
|---|---|---|---|---|---|
| keyword_with_aliases_learned_in_stream | canonical | 1.00 | 0.00 | 0.00 | 0.00 |
| keyword_with_aliases_learned_in_stream | easy_unseen | 0.19 | 0.25 | 0.00 | 0.56 |
| keyword_with_aliases_learned_in_stream | hard_unseen | 0.19 | 0.25 | 0.00 | 0.56 |
| keyword_with_aliases_learned_in_stream | out_of_scope | 0.00 | 0.00 | 1.00 | 0.00 |
| learned | canonical | 0.97 | 0.00 | 0.00 | 0.03 |
| learned | easy_unseen | 0.22 | 0.00 | 0.53 | 0.25 |
| learned | hard_unseen | 0.03 | 0.00 | 0.78 | 0.19 |
| learned | out_of_scope | 0.00 | 0.00 | 0.96 | 0.04 |

## Consolidation

Modules 36 -> 32 after merging functional duplicates (params 48089 -> 43248); mean accuracy 0.971 -> 0.965.
Distillation to 16x16 under a 0.01 gate: 25 installed, 7 rejected; params 43248 -> 33728; held-out 0.987 -> 0.982; world accuracy 0.965 -> 0.965.

| twin pair | agreement | verdict | rules identical in the world | merged |
|---|---|---|---|---|
| boiler_alarm / press_alarm | 1.0 | duplicate | False | True |
| gear_zone / mixer_zone | 1.0 | duplicate | True | True |
| antenna_side / burner_side | 1.0 | duplicate | True | True |
| fuse_grade / chiller_grade | 1.0 | duplicate | True | True |

Archive-by-usage policy: {"fewer_than_2_calls": {"modules": 0, "share_of_traffic": 0.0}, "fewer_than_3_calls": {"modules": 0, "share_of_traffic": 0.0}, "fewer_than_5_calls": {"modules": 0, "share_of_traffic": 0.0}, "fewer_than_10_calls": {"modules": 0, "share_of_traffic": 0.0}}

## Core latency vs module count (base)

| arrival | modules | p50 us | p95 us |
|---|---|---|---|
| 59 | 11 | 1 | 2 |
| 119 | 17 | 3 | 7 |
| 179 | 22 | 1 | 4 |
| 239 | 28 | 2 | 4 |
| 299 | 33 | 1 | 3 |
| 359 | 36 | 2 | 3 |
| 419 | 36 | 2 | 3 |
| 479 | 36 | 1 | 2 |
| 539 | 36 | 2 | 3 |
| 599 | 36 | 2 | 3 |
| 659 | 36 | 2 | 3 |
| 719 | 36 | 2 | 3 |