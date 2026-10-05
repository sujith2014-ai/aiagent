# Validation log for runbooks R1-R5

Result per step: PASS, FAIL (the system misbehaved) or BLOCKED (could not run; not a system failure). Evidence is the command output or file named in the row. The end-to-end path (real Android + real teacher + real OpenClaw) has NOT passed.

## Session 2026-10-05 (research container; no phone, no API keys)

Environment probe: no model API key in the environment; outbound policy denied `api.openai.com`, `duckduckgo.com`, `dl.google.com` (proxy status: "gateway answered 403 to CONNECT"); `api.anthropic.com` and `registry.npmjs.org` reachable; `adb`, Android SDK/NDK absent.

| Runbook step | Result | Evidence |
|---|---|---|
| R3.1 install OpenClaw 2026.6.35 in an isolated dir | PASS | `node openclaw.mjs --version` -> `OpenClaw 2026.6.35 (c283867)` |
| R3.5 `pytest tests/test_phase8.py -k real_openclaw` with `OPENCLAW_BIN` | PASS | `1 passed` |
| R3.3 live search (`live_openclaw_check.py --search-provider duckduckgo`) | BLOCKED | provider listing worked (duckduckgo configured, no key needed); search not performed: `Proxy response (403) !== 200 when HTTP Tunneling`; audit line `result: unavailable:network`; nothing fabricated |
| R3.4 evidence entries with sha256/provenance from a real result | BLOCKED | needs R3.3 |
| R1 live teacher | BLOCKED | no API key in the environment |
| R2 Android device checklist (a)-(g) | BLOCKED | no device, no SDK/NDK (`dl.google.com` denied) |
| R4 on-device learning on a phone | BLOCKED | same as R2 |
| R5 live tool generation, OpenClaw agent turn | BLOCKED | needs R1-class credentials and an OpenClaw model provider |
