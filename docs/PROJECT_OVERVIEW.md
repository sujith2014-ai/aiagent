# Project overview: requirement, what was built, what is proven, what remains

Last updated 2026-10-05. This page is the single place that ties the original requirement to the current state. Details live in the documents it links to; numbers here are copied from committed reports (`benchmarks/reports/`) and findings (`docs/FINDINGS.md`, F1-F56).

## 1. The requirement (context)

**Goal.** A research prototype of a *Self-Growing Modular AI*: one architecture that runs on PC (Windows/Linux/macOS) and Android, not two AIs. Capabilities belong to the system, not to a device. The system should learn new capabilities from a Teacher AI, keep what it learned, package each capability portably, and need the Teacher less over time.

**Hypothesis under test.** A system of small, separately learned, signed capability modules plus a router can keep acquiring capabilities from a teacher, keep what it learned, run the same packages on PC and phone, and depend on the teacher less over time.

**Stack decided up front.** Rust core shared by all platforms; PyTorch for training; ONNX for inference; Android is a Kotlin shell over the Rust core through JNI (no Python/Chaquopy as the final Android core); a Teacher provider abstraction; OpenClaw as the browser/tool/research boundary; capabilities packaged as `.cap` files signed with Ed25519 under explicit trust roots; the Teacher never holds the production signing key.

**Working rules given by the owner.**
- Do not only plan: implement, test, benchmark, document, and continue phase by phase (Phases 0-15, each with a gate: build, test, benchmark, record results and failures, update ADRs, commit).
- Do not claim something works unless it was actually validated. Do not hide negative results; a negative research result is acceptable, faking success is not.
- If blocked by hardware or services: do everything locally validatable, mark the external validation PENDING, give an exact runbook, continue independent work.
- Privacy and policy: minimum-required-context escalation; credentials stay out of model context; the AI cannot grant itself permissions; no unrestricted self-installation of generated code; generated code may not silently raise its own permissions; never commit credentials, keys or large caches.
- Keep documentation synchronised; produce `FINAL_REPORT.md` with the required sections.
- **Architecture correction (latest):** the external teacher is reached through an AI service's *web interface* driven by OpenClaw's browser, with the human doing any login; no model API key is required. The agent must never request, store or automate passwords.

## 2. Architecture in one page

- **Core (Rust, `core/`)**: capability registry (versions, rollback, archive/restore), `.cap` format `cap/1` (zip: signed manifest, hashes, ONNX model, bundled tests, routing hints), a fixed activation sequence (parse, signature, hashes, compatibility, sandbox load, bundled tests, resource check, activate), keyword and learned routers, novelty/calibration/refusal logic (NaN/Inf and unknown inputs are refused, never answered), a bounded execution graph for composition, a default-deny policy engine with operator-only, request-hash-bound approvals, and a pure-Rust ONNX backend (`mlp-lite`, no C toolchain) with `tract` as an optional fallback.
- **Training (Python, `training/`)**: LearningPackage schema; spec-based teaching (teacher returns a restricted rule, a sandboxed evaluator labels data, the *environment* verifies); selective learning order (metadata update, router update, adapt with replay, new module); consolidation (duplicate merge, distillation, pruning, archive); build service that trains, evaluates and signs.
- **Teacher layer (`integrations/teacher/`)**: provider interface, simulator with fault injection, HTTP providers (not run live), and the **browser teacher** (`browser_teacher.py`) that types the minimal request into a chat website through OpenClaw's browser and treats the reply as untrusted data.
- **OpenClaw (`integrations/openclaw/`)**: narrow `infer web search/fetch` adapter and an `openclaw browser` driver; an action broker puts every external action behind the policy engine and an audit log. OpenClaw skills are used (instructions only); plugins are deliberately not.
- **Tool growth (`integrations/toolgrowth/`, `apps/toolhost/`)**: generated Python tools pass compile, static analysis, sandboxed run, the generator's own tests, independent spec cases, properties, routing conflicts, then an operator approval bound to the code digest, before a signed install. The signing key lives in `apps/`, never under `integrations/`.
- **Platforms**: desktop CLI and token-protected loopback app; Android JNI bridge + Kotlin module (tested on the JVM) + an app module; hybrid capability server/client with signed catalog, hash pinning, placement and offline queue; on-device learning in Rust with a device-scoped signing key and server endorsement.

See `docs/ARCHITECTURE.md`, `DESIGN.md`, `docs/CAPABILITY_FORMAT.md`, `docs/SECURITY.md`, `docs/DECISIONS.md` (ADR-001..046).

## 3. Phase status (Phases 0-15)

| Phase | What | Status | Key evidence |
|---|---|---|---|
| 0 | Spec, ADRs, benchmark format | Done | DESIGN.md, DECISIONS.md |
| 1 | Rust core, registry, `.cap`, signing, ONNX | Done | tests, F1 |
| 2 | Training pipeline, LearningPackage, first capability | Done | phase3 report |
| 3 | Three capabilities learned sequentially, regression, restart, constrained profile | Done (simulation) | modular = no forgetting by construction (F1); shared-network baseline collapsed (1.000 to 0.038) |
| 4 | Dynamic routing, reuse, composition graph | Done (hand-written plans) | composition beats end-to-end networks on compositional tasks (count_inside 0.957 vs 0.57, F2); modular is 4-6x slower than a monolithic net (F4) |
| 5 | Unknown/novelty detection, NEEDS_HELP flow, teacher interface | Done with documented failures | keyword routing weak under word overlap (35% fabricated KNOWN on adversarial set, F6); information classification weak (F8) |
| 6 | Spec-based teaching, selective learning, automated evaluation | Done locally | gates reject bad rules; a rule wrong on about 1% of inputs still passes (F10); replay prevents forgetting on domain growth (F12) |
| 7 | Consolidation, learned routing, baselines | Done; small-LLM baseline PENDING | -63% parameters after consolidation (F14); learned router better at paraphrases but costlier (F16) |
| 8 | OpenClaw integration, policy, provenance | Contract-validated against real OpenClaw; live search PENDING | F19-F22 |
| 9 | Desktop app, real datasets | Done | on par with logistic regression/kNN/MLP, not better (F23); novelty rule redesigned (F24) |
| 10 | Android Kotlin shell, JNI, portable inference | Compile + JVM level done | 3 Android targets compile; 16 Kotlin tests; same `.cap` files unchanged (F28, F29) |
| 11 | Hybrid execution, signed delivery, offline | Done on loopback | 6/6 delivery attacks defended; single signing key is a weakness (F32-F35) |
| 12 | On-device learning | Done on PC | accuracy parity with server training (F36); adaptation gated, sometimes refused (F37) |
| 13 | Tool growth with OpenClaw | Done locally, scripted generator | 23 attack candidates all rejected before approval, each sandbox layer alone insufficient (F40-F44) |
| 14 | Long-running benchmark, teacher dependence, growth/consolidation | Done with simulator teacher | 185 teacher calls vs 720 always-asking; matches joint retraining accuracy, no forgetting (F45-F50) |
| 15 | Final evaluation and report | Done | `FINAL_REPORT.md` |
| follow-up | R1-R5 validation attempt, browser teacher, routing and drift work | Mixed, see sections 4-6 | `docs/VALIDATION.md`, F51-F56 |

## 4. What is proven, and the evidence

- **Tests**: 327 passed, 11 skipped by default (338 collected); 10 further tests need `OPENCLAW_BIN` and passed when run separately; Rust 25 unit + 5 ONNX-parity tests; Kotlin/JVM 16 tests on both library variants. The tests found and fixed many real bugs (NaN answered KNOWN, per-call registry writes, tract panics on malformed signed models, early stopping freezing weights, router aliasing polluting routes, an Android compile error, among others: F3, F18, F26, F30, F35, F38, F43, F46, F52).
- **Teacher dependence** (simulator teacher): 185 calls over 723 arrivals vs 720 for always-asking, falling from 39 to about 20 per 100 arrivals, then plateauing; three seeds 183-185 (F45).
- **No forgetting, low training cost**: modular 0.970 mean accuracy vs joint retraining 0.96, replay 0.95, sequential fine-tuning 0.52; modular builds in about an eighth of joint retraining's time; baselines were given oracle routing (F49).
- **Portability**: the same `.cap` files ran unchanged on the PC runtime, a constrained profile (probability difference 0.0), the JVM bridge and the hybrid client; the lite core compiles for aarch64, x86_64 and armv7 Android targets.
- **Security**: policy default-deny, operator-only approvals bound to digests, signed catalogs with replay/freeze/downgrade defences, 23-attack tool-growth matrix, containment of a compromised teacher double (F20, F22, F33, F41).
- **Browser teacher mechanism**: real OpenClaw 2026.6.35 browser drives a *local mock* chat site end to end (learn a capability, login/captcha stops, hostile replies refused, tool generation with approval).

## 5. Known weaknesses and negative results (kept visible)

1. Modular is no more accurate than a conventional small network on single tasks and costs latency and parameters (F1, F4, F49).
2. Routing to **new wordings** is not solved; hard paraphrases need a teacher round trip. The IDF router fixed easy paraphrases but raised fabricated KNOWN on the adversarial set from 0.20 to 0.93; teacher synonym lists made 4 of 36 capabilities unlearnable. Both are opt-in, off by default (F54-F55, ADR-046).
3. **Drift detection** is weak: the original monitor repaired 1 of 9 drifts; CUSUM with capability-isolated probes repaired 3 of 9 (delays 248-342 arrivals, 270 label queries per run, no spurious repairs). Opt-in, off by default (F48, F53).
4. Keyword KNOWN is not trustworthy under ambiguity: 6.3% of served answers were silently wrong in the long run (F47).
5. Security limits: one signing key for catalog and packages in the hybrid server; the CPython tool sandbox is research-grade; static analysis rejects 6 of 10 benign snippets; server endorsement does not prove data honesty (F33, F41, F42, F39).
6. Everything teacher-related used a **simulator teacher** that never errs and knows the world; no real AI service has been used.

## 6. Validation status of the real-world runbooks

| Item | Status |
|---|---|
| OpenClaw install, real-binary contract test, exported-skill loading | PASS |
| OpenClaw live web search (R3) | BLOCKED: the cloud network policy refuses `duckduckgo.com` |
| Browser teacher and tool generator (OpenClaw browser, no API key) | PASS against the local mock site only |
| **R1: real AI website as teacher** | **PENDING**: needs your interactive sign-in on a PC with a display |
| **R5: real-site tool generation / OpenClaw agent turn** | **PENDING** (an agent turn also needs an OpenClaw model provider) |
| Android app Kotlin compile-level check | PASS (partial: framework jar, androidx stubbed); one bug found and fixed |
| **R2: Android SDK/NDK build, APK, link** | **BLOCKED** in the cloud (`dl.google.com` refused); to be done on your PC |
| **R2/R4: real Android phone** (install, execute, latency, battery, thermal) | **PENDING**: no device |
| **True PC to phone test** | **NOT RUN** |

Evidence and exact commands: `docs/VALIDATION.md`, `docs/RUNBOOKS.md` (R1-R5).

## 7. What remains, in order

Decision from the owner: stop modifying the architecture in the cloud; move testing to the Windows PC; do not start another research phase before the real flow has run.

**Milestone A: real teacher on the PC (R1).**
1. On the Windows PC: install Node 22.19+, install OpenClaw (`openclaw@2026.6.35`), start its gateway, start the managed browser with a visible window, open **one** chosen AI website, sign in manually (credentials never go through the agent), open a new empty chat, capture a trimmed `snapshot`.
2. Tune a `SiteAdapter` to that site (message box name, submit key or button, reply selector, busy flag; use `textContent`, and a code-block selector if whitespace matters); the operator sets `terms_acknowledged` after checking the site's terms and approves the URL once with `aicli approve` (action `browser.chat`).
3. Prove: unknown request, the AI does not know, OpenClaw browser asks the real teacher site, the reply is validated and turned into a LearningPackage, a capability is trained and verified against environment examples, signed into a `.cap` and used; then a similar request is answered locally with **no teacher call**. Record outcomes and the reply acceptance rate in `docs/VALIDATION.md`.

**Milestone B: R3 and R5.** Live OpenClaw research with a reachable provider; tool generation through the real site with operator approval; an OpenClaw agent turn only if a model provider is configured.

**Milestone C: Android on the PC and then the phone (R2/R4).** Install JDK 17+, Android SDK (platform 34, build-tools) and NDK r26+; `scripts/build_android_libs.sh`; copy only public trust roots into the app assets; `gradle :app:assembleDebug`; `adb install`. Then the acceptance checklist in RUNBOOKS R2 (library loads and reports backend `onnx-mlp-lite`; the exact PC-built `.cap` installs, SHA-256 equal on both sides; outputs equal the PC's within 1e-5; tampered, untrusted and camera-requiring packages rejected; latency, RSS, battery, temperature over 10 minutes; registry persists after kill and relaunch; airplane mode still answers). Not yet implemented: Android Keystore wrapping of the device key.

**Milestone D: the target demonstration.** PC learns a capability from the real teacher, exports a `.cap`, the phone verifies and installs it, executes it locally; optionally the phone learns a small adaptation and the PC endorses it.

**Only after A-D:** revisit the open research questions (routing to new wordings, drift detection at a realistic label budget, a real teacher's error rate and cost, plan learning, non-numeric tasks, a production sandbox and separated signing roles).

**Windows notes.** Nothing was tried on Windows. The Rust core and Python tests should build there; the tool-growth sandbox uses Linux features (network namespaces, rlimits), so use WSL2 for that part. If native Windows gives trouble, run our code in WSL2 (Windows 11 shows Chromium windows via WSLg).

## 8. How to reproduce in the cloud environment or any Linux box

```
cargo build --release
python3 -m pytest tests -q                                   # default suite
OPENCLAW_BIN=/path/to/openclaw.mjs python3 -m pytest tests/test_browser_teacher.py tests/test_phase13.py tests/test_phase8.py -q -k real
./scripts/run_android_jvm_tests.sh                            # Kotlin/JVM bridge (add LITE=1 for the lite library)
python3 scripts/run_phase14.py --fresh && python3 scripts/summarize14.py     # long-running benchmark, about 45 minutes
python3 scripts/run_routing_study.py                          # routing generalisation study
```
Per-phase experiments: `scripts/run_phase3.py` ... `run_phase13.py`, with `scripts/summarize*.py`.

## 9. Repository map and document index

- `core/`, `platforms/{desktop,android}/`: Rust core, CLI, JNI, Kotlin.
- `training/`, `packages/`, `apps/{desktop,server,hybrid,toolhost}/`: training, packaging, applications.
- `integrations/{teacher,openclaw,toolgrowth}/`: teacher layer, OpenClaw boundary, tool growth.
- `tests/`, `scripts/`, `benchmarks/reports/`: tests, experiments, machine-readable results (`phase3`..`phase14`, `routing_study`).
- Documents: `README.md`, `DESIGN.md`, `FINAL_REPORT.md`, `docs/{ARCHITECTURE, CAPABILITY_FORMAT, LEARNING, SECURITY, ANDROID, OPENCLAW, BENCHMARKS, TESTING, DECISIONS, FINDINGS, STATUS, RUNBOOKS, VALIDATION, PROJECT_OVERVIEW}.md`.

## 10. Rules that continue to apply

Never claim a validation that did not run; mark BLOCKED or PENDING instead. Keep experimental options (IDF router, CUSUM probes, teacher synonyms) off by default unless a measurement justifies enabling them. Never commit credentials, keys or large caches. The agent never requests, stores or types your credentials; interactive logins are always done by you.
