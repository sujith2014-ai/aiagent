# Final report: Self-Growing Modular AI research prototype

Everything below is backed by a test, a committed benchmark report (`benchmarks/reports/`) or a numbered finding (`docs/FINDINGS.md`, F1-F50). Where something was not validated it says **PENDING** or **not validated**, with the runbook in `docs/RUNBOOKS.md`. Nothing here was measured on a phone, with a live LLM teacher, or with live web research.

## 1. What was implemented

| Area | Implemented | Where |
|---|---|---|
| Core (Rust, shared by PC and Android) | Capability registry with versions, rollback, archive/restore; `.cap` format `cap/1` (Ed25519-signed manifest, SHA-256 hashes, bundled tests); fixed activation sequence (parse, signature, hashes, compatibility, sandbox load, bundled tests, resource check, activate); keyword router and learned router; novelty/calibration/refusal logic; bounded execution graph; default-deny policy engine with operator-only approvals; pure-Rust ONNX backend (mlp-lite) plus optional tract | `core/` (about 2,600 lines), `docs/CAPABILITY_FORMAT.md` |
| Training (Python/PyTorch, server side) | LearningPackage schema, spec-based teaching with a sandboxed expression evaluator, build/validation service that signs, selective learning (metadata, router update, adapt with replay, new module), consolidation (duplicates, distillation, pruning, archive) | `training/`, `integrations/` |
| Teacher layer | Provider interface, simulator with fault injection, HTTP providers (not run live), minimum-context escalation, redaction | `integrations/teacher/`, `integrations/escalation.py` |
| OpenClaw | Narrow `infer web search/fetch` adapter validated against a real 2026.6.35 install, broker with policy and audit, skill export validated against the real binary | `integrations/openclaw/`, `integrations/toolgrowth/openclaw_skill.py` |
| Desktop | Token-protected loopback app, CLI (`aicli`) | `apps/desktop/`, `platforms/desktop/` |
| Android | JNI bridge, Kotlin core-bridge module (16 JVM tests), uncompiled app module | `platforms/android/` |
| Hybrid | Capability server, signed catalog, hash pinning, placement, offline queue, device-learned package endorsement | `apps/server/`, `apps/hybrid/` |
| On-device learning | Pure-Rust trainer, ONNX writer, device-scoped signing key, gated adaptation, router update | `core/src/{train_lite,pack,learn}.rs` |
| Tool growth | Generated Python tools through compile, static checks, sandbox, tests, independent spec, operator-bound approval, signed install | `integrations/toolgrowth/`, `apps/toolhost/` |

## 2. Architecture

One Rust core owns registry, validation, routing and execution on every platform. A capability is a small ONNX model in a signed `.cap` package; devices trust only public keys. Android is a Kotlin shell over the core through JNI. Training, teacher calls, research and tool generation happen on PC/server, behind the policy engine; the teacher never holds a signing key; no generated code is installed without an operator approval bound to its digest. See `docs/ARCHITECTURE.md`, `DESIGN.md`, ADR-001..044.

## 3. Test results

- Python: 327 passed, 11 skipped (the skipped ones need `OPENCLAW_BIN` or `RUN_ANDROID_JVM_TESTS=1`; the OpenClaw-dependent ones were run separately and pass, see docs/VALIDATION.md). Rust: 25 unit + 5 ONNX-parity tests. Kotlin/JVM: 16 tests, both library variants.
- The tests found real bugs, all fixed and listed in F3, F18, F26, F30, F35, F38, F43, F46 (for example NaN inputs answered KNOWN, a per-call registry write that cost 10x the compute, tract panics on malformed signed models, early stopping freezing weights, router aliasing polluting routes).

## 4. Benchmark results (what was measured)

- Per-inference core latency 1-4 us (lite backend), wall time per task 20-33 us through the CLI harness; routing glue dominates module compute (F4, F29). Latency stays flat from 11 to 36 modules (F50).
- Package size 2-29 KB per capability; native library 1.3 MB (lite) / 9.1 MB (full) (F31).

## 5. PC results

All phases run on PC. Accuracy on four real datasets through the Rust runtime: iris 1.000, wine 1.000, digits 0.950, tumor 0.965 on small held-out sets, on par with logistic regression, kNN and a scikit-learn MLP, not better (F23). The default novelty rule was redesigned after it refused 4-6% of legitimate rows (F24).

## 6. Android results

- Validated: the core and JNI bridge compile as static libraries for aarch64, x86_64 (both variants) and armv7 (lite only; the full variant fails at the assembler under the zig stand-in). The Kotlin module drives the real core through the real JNI library on the JVM: outputs equal the Rust CLI's within 1e-6, hostile or incompatible packages rejected, 8-thread use, restart, delivery (F28). The learn/adapt/alias bridge calls pass 2 further tests (16 total).
- Compile level, added later: the `app/` module compiled for the first time against the Android 14 framework jar (androidx stubbed) and exposed a real bug, fixed (F52). Not AGP, not an APK, not an NDK link.
- **PENDING (R2, R4):** anything on ART/bionic or a device: loading the `.so`, the app module (never compiled), latency, memory, battery, thermal, NPU, Android Keystore key wrapping.

## 7. Portability results

The same `.cap` files, unchanged, ran on the PC runtime, a resource-constrained profile (1 CPU, 1 GiB address space, 8 MB model limit; maximum probability difference to the full profile 0.0, phase 3), the JVM bridge and the hybrid server/client. No format change was needed. Outputs are bit-identical only within one backend; across tract and mlp-lite they agree within 1e-5 (F29). Portability to real Android hardware is not demonstrated.

## 8. Continual-learning results

- Independent modules cannot forget: that is by construction and uninformative on its own (F1). The informative comparison is the shared network: sequential fine-tuning collapsed (compare_numbers 1.000 to 0.038 in phase 3; mean accuracy 0.52 over 36 tasks in phase 14), a replay buffer reached 0.95, joint retraining 0.96, modular 0.970 at about an eighth of joint retraining's training time (F49). The baselines were given oracle task routing.
- On non-compositional single tasks the modular system shows no accuracy advantage over a conventional tiny network (F1). On compositional tasks with hand-written plans it beats end-to-end networks (count_inside 0.957 vs 0.57) (F2); plans were written by me.
- Domain growth: adaptation with replay extended a capability to a 4x wider range with no loss on the old range, 0 parameters added; without replay the old range dropped 1.000 to 0.962 and the gate rejected it (F12).
- Growth and consolidation: 36 capabilities grew to 48k parameters; merging four functional duplicates and distilling 25 of 32 modules under gates brought it to 33.7k with world accuracy unchanged (0.965); an earlier phase-7 registry shrank 63% in parameters (F14, F50).
- **Negative:** detecting rule drift from sparse feedback mostly failed: none of 3 drifts was repaired at 10-30% feedback, one at 100%, with spurious repairs caused by routing errors (F48). Generalising to new wordings is the weakest part: unseen paraphrases were routed correctly with KNOWN only 19% of the time by the keyword router and 3-22% by the learned router (F50, F16).

## 9. Teacher-dependency results

All with a simulator teacher that never errs.
- Always asking: 720 calls over 723 arrivals; modular system: 185 (about 26%), falling from 39 to about 20 calls per 100 arrivals and plateauing because paraphrases and repeated unsupported requests keep costing calls (F45). Three seeds: 183-185.
- Levers: router updates save 107 calls (292 to 185) at the cost of 13 more wrong answers; a negative cache for refusals saves 38 more (F45).
- Escalating UNCERTAIN routes is essential: serving them cuts calls to 107 but leaves 9 of 36 capabilities never learned (F47). The router-alias bug (F46) alone changed learned-at-first-request from 13/36 to 36/36.
- Earlier phases: related paraphrases cost 3 teacher calls instead of 9 with router updates (F11); a research-led capability needed 2 teacher calls and 1 OpenClaw call, then 9 of 11 follow-ups needed none (F21).
- No real AI service was ever used as teacher (R1): its error rate, cost and latency are unknown, so these numbers measure the gates and loop, not a model. The intended external-teacher path is an AI website driven through OpenClaw's browser with a human doing any login (ADR-045); it works against a local mock only (F51).

## 10. OpenClaw results

- A real OpenClaw 2026.6.35 install validated the `infer` CLI contract and failure modes (F19), and loaded exported tool skills as eligible and model-visible (F44). The policy boundary behaved as specified on 15 requests; a compromised-teacher double was contained 4/4 (F20, F22).
- Tool growth: ten ordinary missing actions were generated (scripted), gated, approved by the operator, installed and used; 5 first attempts were rejected by the generator's own tests or the independent spec (F40). A 23-candidate attack matrix: full gates 23/23 rejected before approval, 0/15 harmful payloads effective; each sandbox layer alone is insufficient (builtins alone failed the dunder escape; the OS layer alone let 12/15 through) (F41).
- **Not validated:** a live web search or fetch success, an `openclaw agent` turn using a skill, a real model as tool generator (R3, R5).

## 11. Security limitations

- Single signing key for catalog and packages in the hybrid server (F33). Android Keystore not used for the device key (F39).
- The CPython audit-hook sandbox is research-grade, not a production boundary; attacks and defences share one author (F41). Static analysis rejects 6 of 10 benign snippets (F42). Approval is only as good as the operator's reading.
- Keyword routing can answer KNOWN for the wrong capability under word overlap (35% on an adversarial set, F6; 6.3% silently wrong in the long run, F47); information classification is a weak placeholder (F8); redaction is regex-based (F9).
- Server endorsement attests a signature, a version and the device's own examples, not honest data (F39). No TLS or WAN in any hybrid test (F32).

## 12. Known failures and negative results

Head-only adaptation cannot repair an input shift (F37); a new module cannot be trained from fewer than 30 examples; ten-sample adaptation made digits worse; the small-LLM baseline and any live-service validation are PENDING; armv7 with the full tract variant does not build here; a capability rule wrong on about 1% of inputs passes the verification gate (F10); one learned router is bigger and slower than the whole keyword router and not clearly safer (F16); drift detection and routing generalisation are open (F48, F50).

## 13. Remaining research questions

1. Does a real LLM teacher produce specs and tools that pass these gates at acceptable cost, and how often does it err quietly (R1, R5)?
2. Can drift be detected without dense feedback (input-shift statistics, probe tasks), and routing errors separated from model drift?
3. Can routing generalise to new wordings without a teacher round trip (a better learned router, embeddings)?
4. What do latency, memory, battery, thermal behaviour and on-device training really cost on phones (R2, R4)?
5. Does the modular advantage survive tasks that are not small numeric classifiers (text, images), where modules must be larger?
6. Can plans be learned instead of written?
7. What replaces the CPython sandbox and the single signing key in a real deployment?

## 14. Does the evidence support or reject the hypothesis?

The hypothesis: a system of small, separately learned, signed capability modules plus a router can keep acquiring capabilities from a teacher, keep what it learned, run the same packages on PC and phone, and need the teacher less over time.

**Supported, within the tested scope:**
- Incremental acquisition without forgetting and without retraining the whole system: yes, at accuracy equal to joint retraining and far above sequential fine-tuning (F1, F49), for small numeric classifiers.
- A portable, signed, verifiable package that runs unchanged across runtimes: yes at the JVM and compile level (F28, F29).
- Decreasing teacher dependence: yes but partial, about 26% of always-asking with a plateau around 20 calls per 100 arrivals, against a simulated, error-free teacher (F45).
- Controlled self-extension (on-device learning, tool growth) with human approval in the path: works as designed and survived the attacks I wrote (F36-F41).

**Not supported or rejected:**
- No accuracy advantage over a conventional small network on single non-compositional tasks (F1, F23); modularity costs 4-6x latency over a monolithic network and about twice the parameters at 36 tasks (F4, F49).
- The system does not yet notice that what it learned went stale (F48), and its routing does not generalise to new wordings (F50). These are the main reasons the teacher keeps being needed.
- The claim that it works on a phone is **untested**, as is everything involving a live LLM or live web research.

Honest summary: the architecture is sound for what it was built to do, and the benchmark found and fixed one serious scale-dependent bug (F46). Its weakest parts are drift detection and routing generalisation. The evidence does not yet say whether it pays off against a real teacher or on real devices; the runbooks R1-R5 list exactly what to run.


## 15. Follow-up after this report: R1-R5 validation attempt, routing and drift work (2026-10-05)

**Validation status (evidence in `docs/VALIDATION.md`). Nothing in R1-R5 beyond what is listed as PASS has been validated in the real world, and the real Android + real teacher + real OpenClaw end-to-end path has NOT passed.**

| Item | Status |
|---|---|
| R3 OpenClaw install, real-binary contract test, exported-skill loading | PASS |
| R3 live web search | BLOCKED: `duckduckgo.com` refused by the network policy even after the allowlist change (re-verified in two sessions) |
| Browser-based teacher and tool generator (OpenClaw browser, no API key) | PASS against a local mock chat site only (7 real-browser tests); NOT validated against any real AI website |
| R1 real teacher | PENDING: the chat sites are unreachable from the container and a real site needs your own interactive sign-in on a machine with a display |
| R2 / R4 Android SDK, NDK, APK, link | BLOCKED: `dl.google.com` refused |
| R2 / R4 physical phone | PENDING: no device; nothing claimed |
| R5 OpenClaw agent turn, real-site tool generation | PENDING |

**Reproduction.** The complete Phase 14 suite was rerun after the changes below: all 8 original arms reproduced their earlier numbers exactly (for example base 185 teacher calls, 41 wrong answers).

**Routing generalisation (F54, F55): not solved; nothing lexical helps hard paraphrases.** An IDF-weighted router lifted easy paraphrases from 0% to 100% KNOWN-correct in a 36-capability registry, but on the 3-capability adversarial set it raised fabricated KNOWN from 0.20 to 0.93, and end to end only 21 of 36 new capabilities were learned at first request (vs 36). Teacher-supplied synonym lists caused 4 of 36 capabilities to never be learned and 31% wrong-KNOWN on hard paraphrases. Both stay opt-in; the keyword router remains the default (ADR-046).

**Drift detection (F53): improved but insufficient.** A CUSUM test over capability-isolated probes with 270 label queries per run repaired 3 of 9 drifts across three seeds (delays 248-342 arrivals, no spurious repairs) against 1 of 9 for the original window monitor (2 spurious repairs). It misses most drifts at a realistic budget and did not reliably reduce wrong answers, so it stays opt-in.

**Effect on the verdict (section 14).** Unchanged: the two weakest parts, routing to new wordings and noticing stale capabilities, remain open, and the claims that need a real teacher, a real website or a phone remain untested.
