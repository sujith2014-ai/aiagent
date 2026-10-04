# Phase 6 benchmark summary (2026-10-04, Linux-6.18.44-fc-v64-x86_64-with-glibc2.39)

## Teacher rule fault matrix (spec mode; environment-verified at 0.95)

| fault | compare_numbers | point_region | argmax_position |
|---|---|---|---|
| None | ANSWER | ANSWER | ANSWER |
| contradicts_examples | NEEDS_HELP | NEEDS_HELP | NEEDS_HELP |
| malicious_expr | NEEDS_HELP | NEEDS_HELP | NEEDS_HELP |
| plausible_wrong | NEEDS_HELP | NEEDS_HELP | NEEDS_HELP |
| subtle_wrong | ANSWER | NEEDS_HELP | NEEDS_HELP |

`subtle_wrong` is only subtle for compare_numbers (wrong on ~1% of inputs, accepted); the other two use the gross faults.

## Provider path (mock HTTP endpoints)

- openai_compat_end_to_end: {'result': 'ANSWER', 'path': ['local', 'teacher', 'learn'], 'requests': 1, 'auth_header_sent': True, 'request_body_fields': ['attempted_route', 'available_tools', 'evidence', 'failure', 'input_dim', 'known_capabilities', 'task_intent']}
- anthropic_style_end_to_end: {'result': 'ANSWER', 'path': ['local', 'teacher', 'learn'], 'api_key_header': True}
- server_error_503: {'result': 'QUEUED_OFFLINE', 'teacher_calls_counted': 0, 'queued': True}
- garbage_text_response: {'result': 'NEEDS_HELP', 'capabilities_installed': 0}
- live_endpoint_validation: PENDING: requires network egress and an API key (see docs/RUNBOOKS.md)

## Paraphrase handling: router_update vs per-task rerouting (36 paraphrase tasks, 3 rounds)

| mode | teacher calls | per round | accepted router updates | answered |
|---|---|---|---|---|
| without_router_update | 9 | [3, 3, 3] | 0 | 36/36 |
| with_router_update | 3 | [3, 0, 0] | 3 | 36/36 |

## Domain extension (compare_numbers must work on a 4x wider range)

Before: `OUT_OF_DISTRIBUTION`. Chosen strategy: **adapt_existing**.

| strategy | new-domain verification | old-domain before -> after | params added | passed |
|---|---|---|---|---|
| metadata_update | 0.943 |  ->  |  | False |
| adapt_existing | 0.993 | 1.000 -> 1.000 | 0 | True |

Ablation (not installed): naive fine-tune without replay: new 0.990, old 1.000 -> 0.962 (regression: old-domain accuracy 1.000 -> 0.962).
After extension: {'result': 'ANSWER', 'label': 'GREATER', 'expected': 'GREATER'}; bundled regression 1.0.

## Conflicting rule change (point_region now means a smaller circle)

Chosen strategy: **new_module**.

| strategy | new-domain verification | old-domain before -> after | params added | passed |
|---|---|---|---|---|
| metadata_update | 0.760 |  ->  |  | False |
| adapt_existing | 0.937 | 0.992 -> 0.851 | 0 | False |
| new_module | 0.993 |  ->  | 1218 | True |

Routing afterwards: {'old_intent': {'capability': 'point_region', 'status': 'KNOWN', 'label': 'INSIDE'}, 'new_intent': {'capability': 'point_region__ext', 'status': 'KNOWN', 'label': 'OUTSIDE'}}. Probe: (0.5, 0.0) has r^2=0.25: INSIDE under the old rule (r^2<0.5), OUTSIDE under the new rule (r^2<0.2)
