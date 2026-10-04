# Runbooks for validation that could not be executed in the research container

## R1. Live teacher (PENDING)
Blocked here by: no API credentials and restricted network egress. Everything up to the HTTP boundary is tested against a local mock server (`tests/test_phase6.py`).
To validate against a real model:
1. `export OPENAI_API_KEY=...` (or `ANTHROPIC_API_KEY`). Qwen works through any OpenAI-compatible endpoint (DashScope compatible mode, vLLM, Ollama).
2. `python3 scripts/live_teacher_check.py openai-compat https://<host>/v1 <model>` or `... anthropic <model>`.
3. Expected: `result: ANSWER`, `path: [local, teacher, learn]`. Acceptable failures are `NEEDS_HELP` with a reason from spec validation (the model produced a malformed or wrong rule); these are data about the model, not bugs. Record the model name, date and outcome in docs/FINDINGS.md.
4. Repeat 20 times per task (non-zero temperature models vary) and record the acceptance rate; the fault matrix in benchmarks/reports/phase6.json shows what the gates catch.

## R2. Android (PENDING, see docs/ANDROID.md) and R3. OpenClaw (PENDING, see docs/OPENCLAW.md)
Added when those phases are reached.
