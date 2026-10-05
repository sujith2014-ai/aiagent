# Runbooks for validation that could not be executed in the research container

## R1. Live teacher (PENDING)
**Primary path (ADR-045): a chat website through OpenClaw's browser, no API key.** On a machine with a display and the target site reachable: install OpenClaw, start its gateway, `openclaw browser --browser-profile openclaw start`, open the site and sign in yourself, set `SiteAdapter(terms_acknowledged=True, ...)` after checking the site's terms, approve the URL with `aicli approve` (action `browser.chat`, params `{"url": ...}`), then run the escalation with `BrowserTeacher`. Record outcomes in docs/VALIDATION.md. The API steps below are an optional alternative.
Blocked here by: no API credentials and restricted network egress. Everything up to the HTTP boundary is tested against a local mock server (`tests/test_phase6.py`).
To validate against a real model:
1. `export OPENAI_API_KEY=...` (or `ANTHROPIC_API_KEY`). Qwen works through any OpenAI-compatible endpoint (DashScope compatible mode, vLLM, Ollama).
2. `python3 scripts/live_teacher_check.py openai-compat https://<host>/v1 <model>` or `... anthropic <model>`.
3. Expected: `result: ANSWER`, `path: [local, teacher, learn]`. Acceptable failures are `NEEDS_HELP` with a reason from spec validation (the model produced a malformed or wrong rule); these are data about the model, not bugs. Record the model name, date and outcome in docs/FINDINGS.md.
4. Repeat 20 times per task (non-zero temperature models vary) and record the acceptance rate; the fault matrix in benchmarks/reports/phase6.json shows what the gates catch.

## R2. Android on a physical device (PENDING, see docs/ANDROID.md)
Blocked here by: no Android SDK/NDK (dl.google.com unreachable), no device.
1. Install JDK 17+, Android SDK (platform 34, build-tools) and NDK r26+; set `ANDROID_HOME` and `ANDROID_NDK_HOME`.
2. `scripts/build_android_libs.sh` (builds the lite core for arm64-v8a, armeabi-v7a, x86_64 into `app/src/main/jniLibs/`; needs the NDK only to link).
3. Copy the public trust roots (`{"key_id": "<base64 public key>"}`) to `platforms/android/app/src/main/assets/trust.json`. Never put a private key on the device.
4. `cd platforms/android && gradle :app:assembleDebug` and `adb install -r app/build/outputs/apk/debug/app-debug.apk`. Fix compile errors in `app/` first: those Kotlin files have never been compiled.
5. Serve a PC-built `.cap` over HTTPS (the app forbids cleartext; use a real or locally trusted certificate, or temporarily relax `network_security_config.xml` for a debug build only).
6. Acceptance checklist, recording results in docs/FINDINGS.md: (a) logcat shows the library loads and `nativeVersion` reports backend `onnx-mlp-lite`; (b) the app installs the exact PC-built `.cap` (compare SHA-256 on both sides); (c) solve outputs equal the PC's within 1e-5 for the cases in `runs/android_fixtures/expectations.json`; (d) tampered, untrusted and camera-requiring packages are rejected; (e) latency per inference and RSS (`adb shell dumpsys meminfo`), battery (`dumpsys batterystats`) and temperature over a 10-minute loop; (f) kill the process and relaunch: the registry persists; (g) airplane mode: installed capabilities still answer.
7. If any step needs a change to the package format or capability semantics, that is a portability failure: investigate before continuing.

## R3. Live OpenClaw research (PENDING)
Blocked here by: egress proxy denies the search provider (HTTP 403 on CONNECT), no web/model provider credentials, Node 22 only (latest OpenClaw needs Node >= 24.16; 2026.6.35 works on Node >= 22.19).
1. Install in an isolated directory: `npm install openclaw@2026.6.35 --ignore-scripts` (or the latest on Node 24+), then `node node_modules/openclaw/openclaw.mjs --version`.
2. Allow outbound HTTPS to the chosen search provider (DuckDuckGo needs no key: `infer web providers --json` lists it as configured; others need keys, e.g. GEMINI_API_KEY, set in the OpenClaw profile state dir, not in this repo).
3. `OPENCLAW_BIN=/path/to/openclaw.mjs python3 scripts/live_openclaw_check.py --search-provider duckduckgo --query "how to compare two numbers"` and, for fetch, add the page's domain to `integrations/openclaw/policy.default.json` and pass `--fetch-url`.
4. Expected: provider listing, evidence entries with sha256/provenance, audit log lines. If the success JSON does not parse into evidence, the adapter raises `no_results`/`bad_output` (not fabricated evidence): adjust `_hits` in `integrations/openclaw/openclaw_cli.py` to the real provider shape and add a contract test with the captured output.
5. Run `OPENCLAW_BIN=... python3 -m pytest tests/test_phase8.py -q -k real_openclaw` and record results (and the OpenClaw version) in docs/FINDINGS.md.
6. For a real teacher via OpenClaw: configure model auth in the OpenClaw profile, then use `OpenClawModelTeacher` (not validated).


## R4. On-device learning on a real phone: battery, thermal, NPU (PENDING)
Blocked here by: no Android device, no NDK. Everything below the physical layer is validated (F36-F38): the Rust trainer, ONNX writer, signing, gates, JNI/Kotlin bridge on the JVM.
1. Build the lite JNI library and the CLI for the phone ABI: `scripts/build_android_libs.sh` (needs `ANDROID_NDK_HOME`); for a quick check also cross-build `aicli` (`cargo build --release -p aicli --target aarch64-linux-android --no-default-features` with the NDK linker configured) and `adb push` it with a `learn` spec (see tests/test_phase12.py for the shape) and a key from `aicli keygen`.
2. Run the 4 tasks of scripts/run_phase12.py on the phone: `adb shell taskset -c <big core> ./aicli --root /data/local/tmp/rt --trust trust.json --device-key key.json learn --spec spec.json`. Record train_seconds, peak_rss_kb (from the report) and wall time, five repeats, screen off, airplane mode, battery 100% -> note `dumpsys battery` level before/after a 200-run loop.
3. Thermal: sample `/sys/class/thermal/thermal_zone*/temp` and CPU frequency each second during a 10-minute continuous learn loop; report throttling onset and the slowdown relative to the first minute.
4. NPU/GPU: not used. The trainer is CPU-only and inference is the mlp-lite CPU backend; NNAPI/LiteRT would need a separate backend (not built). Do not claim NPU results.
5. Keystore: wrap the 32-byte seed with an Android Keystore AES key in the Kotlin shell before `setDeviceKey` (not implemented).
6. Compare against F36 (PC, one pinned core) and record the phone numbers and device model in docs/FINDINGS.md.


## R5. Live tool generation and an OpenClaw agent turn (PENDING)
Blocked here by: no model credentials or egress for a generator, no OpenClaw model provider. Validated: gates, sandbox, approval, installation, skill export and loading by the real OpenClaw 2026.6.35.
1. Implement `ToolGenerator.generate` (integrations/toolgrowth/generator.py) over a `TeacherProvider`: send only the `ToolRequest` (description, visible examples, the allowed-module list and the `run(inp)` contract) and, on retries, `Report.feedback()`. Never send the hidden spec cases, credentials or registry contents. Redact as for the Teacher (R1).
2. Run `scripts/run_phase13.py` with that generator in place of `ScriptedGenerator` for the ten tasks, 5 repeats; record per-task attempts, the stage that rejected each attempt, how often the model's code needs a module outside the allowlist, and the operator's time to review. Report rejected-by-static rates honestly (F42 predicts many).
3. OpenClaw: configure a model provider in the OpenClaw profile (not in this repo), install the exported skill (`openclaw skills install <dir>`), run `openclaw agent --message "..."`, and check from the audit log that the agent called `scripts/aitool.py` and nothing else, that a denied or refused call was not retried around the host, and that it never tried to install or approve a tool.
4. Before any production use replace the CPython sandbox with an OS-level one (seccomp/gVisor/WASM) and re-run the attack matrix.
