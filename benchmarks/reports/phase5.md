# Phase 5 benchmark summary (2026-10-04, Linux-6.18.44-fc-v64-x86_64-with-glibc2.39)

## Calibration (temperature scaling fit on validation, ECE on held-out)

| capability | T | ECE before | ECE after |
|---|---|---|---|
| compare_numbers | 0.050 | 0.0582 | 0.0063 |
| point_region | 0.458 | 0.0103 | 0.0034 |
| argmax_position | 0.050 | 0.0238 | 0.0118 |

T=0.050 is the lower edge of the search grid (the model is under-confident); see FINDINGS.

## Unknown/novelty detection: signal ablation

Rates are fractions of each set. `fabrication` = out-of-scope/OOD task answered with status KNOWN (lower is better).

| set | metric | keyword | novelty | confidence | calibrated |
|---|---|---|---|---|---|
| S1_in_scope_canonical | answered_known | 1.000 | 1.000 | 0.968 | 1.000 |
| S1_in_scope_canonical | accuracy_known_only | 0.992 | 0.992 | 0.992 | 0.992 |
| S1_in_scope_canonical | accuracy_all_answers | 0.992 | 0.992 | 0.992 | 0.992 |
| S1_in_scope_canonical | needs_help | 0.000 | 0.000 | 0.000 | 0.000 |
| S2_in_scope_paraphrase | answered_known | 0.594 | 0.594 | 0.572 | 0.584 |
| S2_in_scope_paraphrase | accuracy_known_only | 0.989 | 0.989 | 1.000 | 0.995 |
| S2_in_scope_paraphrase | accuracy_all_answers | 0.992 | 0.992 | 0.992 | 0.992 |
| S2_in_scope_paraphrase | needs_help | 0.250 | 0.250 | 0.250 | 0.250 |
| S3_in_scope_intent_ood_input | fabrication_rate | 1.000 | 0.000 | 0.000 | 0.000 |
| S3_in_scope_intent_ood_input | answered_uncertain | 0.000 | 0.000 | 0.000 | 0.000 |
| S3_in_scope_intent_ood_input | needs_help | 0.000 | 1.000 | 1.000 | 1.000 |
| S4_out_of_scope | fabrication_rate | 0.067 | 0.067 | 0.067 | 0.067 |
| S4_out_of_scope | answered_uncertain | 0.133 | 0.133 | 0.133 | 0.133 |
| S4_out_of_scope | needs_help | 0.800 | 0.800 | 0.800 | 0.800 |
| S5_adversarial_keyword_overlap | fabrication_rate | 0.350 | 0.350 | 0.333 | 0.350 |
| S5_adversarial_keyword_overlap | answered_uncertain | 0.650 | 0.650 | 0.667 | 0.650 |
| S5_adversarial_keyword_overlap | needs_help | 0.000 | 0.000 | 0.000 | 0.000 |
| S6_boundary_points | answered_known | 1.000 | 1.000 | 0.850 | 0.947 |
| S6_boundary_points | accuracy_known_only | 0.923 | 0.923 | 0.976 | 0.951 |
| S6_boundary_points | accuracy_all_answers | 0.923 | 0.923 | 0.923 | 0.923 |
| S6_boundary_points | needs_help | 0.000 | 0.000 | 0.000 | 0.000 |

## Escalation over a task stream

Teacher calls: 9 over 19 tasks. Related re-encounters answered locally without the teacher: 10/12.
Offline: unknown task -> QUEUED_OFFLINE (teacher called while offline: False); known capability still answers offline: True.

| kind | intent | result | path | teacher called |
|---|---|---|---|---|
| first | compare these two numbers | ANSWER | local > teacher > learn | True |
| first | is this point inside the circular region | ANSWER | local > teacher > learn | True |
| first | find the position of the largest value | ANSWER | local > teacher > learn | True |
| related | compare these two numbers | ANSWER | local | False |
| related | is this point inside the circular region | ANSWER | local | False |
| related | find the position of the largest value | ANSWER | local | False |
| related | order the two values | ANSWER | local > teacher > reroute | True |
| related | classify the point as inside or outside | ANSWER | local | False |
| related | pick the winning slot | ANSWER | local > teacher > reroute | True |
| related | is the first one greater than, smaller than or the same as the second | ANSWER | local | False |
| related | locate point relative to the circular region | ANSWER | local | False |
| related | position of the highest value | ANSWER | local | False |
| unknown | translate this sentence to french | NEEDS_HELP | local > teacher | True |
| research | what is the weather today | NEEDS_EXTERNAL | local > teacher | True |
| tool | send email to my colleague | NEEDS_EXTERNAL | local > teacher | True |
| memory | remember where my car is parked | MEMORY_UPDATED | local > teacher > memory | True |
| related | compare these two numbers | ANSWER | local | False |
| related | is this point inside the circular region | ANSWER | local | False |
| related | find the position of the largest value | ANSWER | local | False |

## Hostile / malformed teacher output

| case | result | capabilities installed afterwards |
|---|---|---|
| not_json | NEEDS_HELP | 0 |
| unknown_action | NEEDS_HELP | 0 |
| bad_learning_package | NEEDS_HELP | 0 |
| reroute_to_missing | NEEDS_HELP | 0 |

## Privacy filter

Request actually sent: `{"task_intent": "compare numbers for user [EMAIL] [SECRET] token [SECRET] in [PATH]", "input_dim": 2, "known_capabilities": ["compare_numbers"], "attempted_route": {"status": "NO_MATCH", "candidates": []}, "failure": "", "available_tools": [], "evidence": []}`
Leaked secrets: []

## Information classification (rule baseline)

Tuned set: 1.00 on 20 items (written together with the rules: not evidence).
Harder set: 0.50 on 10 items; errors: [['Teach yourself to tell sarcasm from sincerity', 'CAPABILITY', 'KNOWLEDGE'], ['The sort function in this library is stable', 'KNOWLEDGE', 'CAPABILITY'], ['Improve at spotting fraudulent invoices', 'CAPABILITY', 'KNOWLEDGE'], ['I need the tool to predict churn', 'CAPABILITY', 'KNOWLEDGE'], ['Learn which emails are urgent', 'CAPABILITY', 'KNOWLEDGE']]
