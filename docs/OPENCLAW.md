# OpenClaw integration (Phase 8)

## What OpenClaw is here
The npm package `openclaw` (MIT; "Multi-channel AI gateway", a personal-assistant gateway with channels, skills, plugins, an exec-approval system and a CLI). Facts established by reading the package and running it:
- npm lists 2026.9.8 as latest (needs Node >= 24.16 or >= 26.1) but `npm pack openclaw` returned 2026.6.35 (needs Node >= 22.19). The discrepancy is unexplained. **Validation here used 2026.6.35 on Node 22.22**, installed in an isolated directory with install scripts disabled. The PyPI package `openclaw` (2.0.2) was not examined and is not used.
- The unpacked package is ~389 MB with Node dependencies: **it cannot run on the Android device; OpenClaw is a PC/server component** (Phase 10/11 consequence).
- Real CLI surface used: `openclaw [--profile P] infer web search --query Q [--limit N] [--provider ID] --json`, `infer web fetch --url U --json`, `infer web providers --json`, and (optionally) `infer model run --local --model M --prompt P --json`. Success: exit 0, JSON on stdout (envelope `{"ok":true,"capability":..,"provider":..,"outputs":[{"result":..}]}`, from OpenClaw's source). Failure: exit 1, empty stdout, message on stderr.

## Boundary (ADR-019..022)
```
Escalator -> ActionBroker -> policy (Rust, deterministic) -> OpenClawActionProvider -> subprocess `openclaw infer web ...`
                 |-> audit log (every decision, incl. denials)        isolated HOME/profile, no inherited secrets
```
- We use the narrow, non-agentic `infer` commands. We never start `openclaw agent` turns (autonomous, own tools), never use its shell/browser/files/channels, and never hand it our registry, packages or credentials.
- All external actions pass the policy layer first; a denied or approval-pending action never reaches the provider. Unavailable providers (not configured, network blocked, timeout, bad output) produce `QUEUED_EXTERNAL` with the real reason; nothing is ever fabricated.
- Evidence is untrusted data with provenance (`provider`, `source`, `retrieved_at`, `sha256`, `simulated`). It is stored in the signed package manifest and is never promoted without environment verification.

## Status
| Item | Status |
|---|---|
| Install + CLI contract + failure modes against real binary | validated (tests/test_phase8.py::test_real_openclaw_binary_contract with `OPENCLAW_BIN`) |
| Search/fetch **success** parsing | source-derived envelope, defensive inner parsing; **not validated live** |
| Live web research | **PENDING** (egress proxy returns 403 for the search provider; no provider credentials) |
| `infer model run` as a teacher channel | implemented, **not validated** (needs model auth) |
| `openclaw agent`, gateway, channels | not used |
| Skills (Phase 13) | approved tools exported as `SKILL.md`; real 2026.6.35 installs them and lists them eligible and model-visible; **agent turn using one PENDING (R5)** |
| Plugins | deliberately not used (ADR-041) |
Runbook: docs/RUNBOOKS.md R3.


## Tool growth and OpenClaw (Phase 13)
OpenClaw skills are markdown files (`name`, `description`, optional single-line `metadata` JSON for gating) that teach the agent how to use tools; `openclaw skills install ./dir`, `list --json`, `info` and `check` work offline. Our integration exports an approved tool as a skill that tells the agent to run `scripts/aitool.py --root <registry> run <tool> --input <json>` and nothing else; that host enforces policy, signature/hash verification and the sandbox, so the skill grants no extra power. Skills are preferred over plugins because a plugin runs inside OpenClaw with its own permissions. See FINDINGS F40-F44, ADR-038..042 and docs/RUNBOOKS.md R5.
