# Runbooks for validation that could not be executed in the research container

## R1. Live teacher (PENDING)
Blocked here by: no API credentials and restricted network egress. Everything up to the HTTP boundary is tested against a local mock server (`tests/test_phase6.py`).
To validate against a real model:
1. `export OPENAI_API_KEY=...` (or `ANTHROPIC_API_KEY`). Qwen works through any OpenAI-compatible endpoint (DashScope compatible mode, vLLM, Ollama).
2. `python3 scripts/live_teacher_check.py openai-compat https://<host>/v1 <model>` or `... anthropic <model>`.
3. Expected: `result: ANSWER`, `path: [local, teacher, learn]`. Acceptable failures are `NEEDS_HELP` with a reason from spec validation (the model produced a malformed or wrong rule); these are data about the model, not bugs. Record the model name, date and outcome in docs/FINDINGS.md.
4. Repeat 20 times per task (non-zero temperature models vary) and record the acceptance rate; the fault matrix in benchmarks/reports/phase6.json shows what the gates catch.

## R2. Android (PENDING, see docs/ANDROID.md)
Added when that phase is reached.

## R3. Live OpenClaw research (PENDING)
Blocked here by: egress proxy denies the search provider (HTTP 403 on CONNECT), no web/model provider credentials, Node 22 only (latest OpenClaw needs Node >= 24.16; 2026.6.35 works on Node >= 22.19).
1. Install in an isolated directory: `npm install openclaw@2026.6.35 --ignore-scripts` (or the latest on Node 24+), then `node node_modules/openclaw/openclaw.mjs --version`.
2. Allow outbound HTTPS to the chosen search provider (DuckDuckGo needs no key: `infer web providers --json` lists it as configured; others need keys, e.g. GEMINI_API_KEY, set in the OpenClaw profile state dir, not in this repo).
3. `OPENCLAW_BIN=/path/to/openclaw.mjs python3 scripts/live_openclaw_check.py --search-provider duckduckgo --query "how to compare two numbers"` and, for fetch, add the page's domain to `integrations/openclaw/policy.default.json` and pass `--fetch-url`.
4. Expected: provider listing, evidence entries with sha256/provenance, audit log lines. If the success JSON does not parse into evidence, the adapter raises `no_results`/`bad_output` (not fabricated evidence): adjust `_hits` in `integrations/openclaw/openclaw_cli.py` to the real provider shape and add a contract test with the captured output.
5. Run `OPENCLAW_BIN=... python3 -m pytest tests/test_phase8.py -q -k real_openclaw` and record results (and the OpenClaw version) in docs/FINDINGS.md.
6. For a real teacher via OpenClaw: configure model auth in the OpenClaw profile, then use `OpenClawModelTeacher` (not validated).
