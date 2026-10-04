# Phase 11 benchmark summary (2026-10-04, Linux-6.18.44-fc-v64-x86_64-with-glibc2.39)

Network: loopback only: latencies are lower bounds; no TLS; no real WAN.

## Server-side training and signed delivery (4 real tasks, constrained device profile)

| task | learned | held-out acc | upload (JSON bytes) | package bytes | server train+sign s | device verify+install s | installed |
|---|---|---|---|---|---|---|---|
| iris_species | True | 1.0 | 3943 | 20010 | 1.56 | 0.009 | True |
| wine_cultivar | True | 1.0 | 15105 | 9599 | 0.17 | 0.009 | True |
| digit_recognition | True | 0.9722222089767456 | 615330 | 28985 | 1.07 | 0.012 | True |
| tumor_malignancy | True | 0.9736841917037964 | 139858 | 21493 | 0.46 | 0.015 | True |

Signed catalog: 1633 bytes. Device holds a private key: **False**.

## Local vs remote execution

Remote (loopback HTTP, server runs the core per request): p50 4881 us, p95 6880 us. Local, including the harness's process spawn per call: p50 3642 us. local numbers include spawning the CLI process per call (the harness), so they overstate on-device latency; core inference alone is ~1-4 us (benchmarks/reports/backend_bench.json)

## Placement

- **known_task**: {"placement": "LOCAL", "result": "ANSWER", "executed_on": "device"}
- **unknown_task_online**: {"placement": "SERVER", "result": "NEEDS_HELP", "executed_on": "server"}
- **sensitive_unknown_task**: {"placement": "REFUSE_KEEP_LOCAL", "result": "NEEDS_HELP", "server_requests_made": 0}
- **sensitive_examples_upload**: {"uploaded": false, "reason": "examples are sensitive or upload was not consented to; nothing left the device", "server_requests_made": 0}

## Offline timeline

Known tasks answered while offline: 3/3; unknown tasks queued (not pretended): 2; teach while offline: {'uploaded': False, 'queued': True}; pending before reconnect: 3; flush while still offline: {'flushed': 0, 'still_queued': 3}; flush after reconnect: {'flushed': 3, 'still_queued': 0}.

## Attacks on the delivery path

| attack | defended | outcome | capabilities on device afterwards |
|---|---|---|---|
| tampered_catalog | True | {"refused": "catalog signature invalid"} | [] |
| catalog_signed_by_unknown_key | True | {"refused": "catalog signed by untrusted or revoked key 'evil'"} | [] |
| modified_package_in_transit | True | {"installed": [], "rejected": ["downloaded bytes do not match the signed catalog entry"], "warnings": []} | [] |
| catalog_replay | True | {"refused": "catalog replay: sequence 1 < last seen 2"} | ['iris_species'] |
| stale_catalog_freeze | True | {"refused": "catalog is older than the maximum allowed age (possible freeze attack)"} | [] |
| downgrade_via_signed_catalog | True | {"installed": [], "rejected": [], "warnings": ["iris_species: catalog offers 0.1.0 but 0.2.0 is installed (possible downgrade attempt); igno | ['iris_species'] |