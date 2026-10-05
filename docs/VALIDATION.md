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


## Session 2026-10-05, part 2 (after the network allowlist change and the browser-teacher architecture correction)

Architecture correction recorded: the external teacher is reached through an AI service's WEB interface using OpenClaw's browser tool, not through an API key. Gap found: before this change the integration only had HTTP API providers (`integrations/teacher/http_providers.py`) and no browser path. Closed by `integrations/teacher/browser_teacher.py`, `integrations/openclaw/browser.py` (driver over the supported `openclaw browser` CLI) and `integrations/toolgrowth/browser_generator.py` (ADR-045).

Network after the change: `duckduckgo.com`, `html.duckduckgo.com`, `dl.google.com` and the AI chat sites tried (`chatgpt.com`, `gemini.google.com`, `chat.qwen.ai`, `chat.deepseek.com`) are STILL refused by the proxy (403 on CONNECT, see `$HTTPS_PROXY/__agentproxy/status`); `claude.ai` returns 403; Gradle hosts and `maven.google.com` answer, but `maven.google.com` redirects to `dl.google.com`, and Maven Central answered 429. The added domains did not take effect in this running session.

| Step | Result | Evidence |
|---|---|---|
| R3.3 live search (rerun) | BLOCKED | `Proxy response (403) !== 200 when HTTP Tunneling`, audit `unavailable:network` |
| OpenClaw gateway + managed browser start headless here (Chromium from /opt/pw-browsers) | PASS | `browser doctor`: gateway reachable, plugin enabled, profile openclaw; `browser [openclaw] running: true (headless)` |
| Browser teacher against a LOCAL MOCK chat site, through the REAL OpenClaw browser: reply obtained, minimal request typed, full escalation learns a capability (`path [local, teacher, learn]`, second request fully local) | PASS | tests/test_browser_teacher.py (real-browser tests, `OPENCLAW_BIN` set) |
| Login wall and captcha: stops, types nothing, names the manual action, request queued | PASS (mock) | same file; `TeacherNeedsHuman` text "sign in yourself ... do not give credentials" |
| Garbage reply and prompt-injection reply install nothing | PASS (mock) | same file |
| Browser access requires operator approval bound to the URL; terms acknowledgement required | PASS | unit tests with the real policy engine |
| Tool generation through the browser (R5 without a model key): buggy first attempt rejected by the gates, generic feedback only, hidden spec cases never sent, operator approval, install, use | PASS (mock) | `test_real_end_to_end_tool_growth_through_the_browser_generator_and_operator_approval` |
| Browser teacher/generator against a REAL AI website | BLOCKED | sites unreachable from this container; also needs the operator to sign in by hand in the OpenClaw browser profile on a machine with a display (the container is headless); the site's terms must permit automated use |
| R1 real teacher (browser path) | BLOCKED | same as above |
| R5 OpenClaw agent turn | BLOCKED | `openclaw agent` needs a configured model provider; the browser-teacher path replaces it for generation, an agent turn is not claimed |
| R2/R4 Android SDK/NDK build | BLOCKED | `dl.google.com` refused, Google Maven redirects there, Maven Central 429; nothing from the SDK, NDK or AGP could be downloaded. Already passing from earlier: lite JNI libraries compile for 3 Android targets, Kotlin/JVM bridge 16 tests |
| R2/R4 physical phone | NOT RUN | no device; not claimed |

Findings from this work: OpenClaw's managed browser blocks private-network URLs by default (`browser.ssrfPolicy`), so the mock tests enable `dangerouslyAllowPrivateNetwork` in a throwaway test profile only; managed Chrome launches direct (no `HTTPS_PROXY`), so inside a proxied container it needs `browser.extraArgs --proxy-server=...`; reading replies with `innerText` collapsed runs of spaces and destroyed code indentation inside JSON strings, so the default extractor uses `textContent` (sites that render code in `<pre>` need an adapter that targets it); every browser action is a separate CLI process, so a full chat round trip takes tens of seconds.

## Session 2026-10-05, part 3 (new session; network re-verified first)

Network check (curl through `$HTTPS_PROXY`, status page `recentRelayFailures`): `duckduckgo.com`, `html.duckduckgo.com`, `dl.google.com`, `chatgpt.com`, `gemini.google.com`, `chat.qwen.ai`, `chat.deepseek.com` still 403 on CONNECT; `claude.ai` 403; reachable: `maven.google.com` (301 to `dl.google.com`), Maven Central (200), `plugins.gradle.org`, `services.gradle.org`, `registry.npmjs.org`, PyPI. The requested allowlist entries for DuckDuckGo and Google were NOT in effect for this session.

| Step | Result | Evidence |
|---|---|---|
| R3.3 live search rerun | BLOCKED | still `Proxy response (403) !== 200 when HTTP Tunneling`; audit `unavailable:network`; nothing fabricated |
| Android SDK / NDK / AGP install | BLOCKED | `dl.google.com` refused; no NDK source reachable |
| Android `app/` Kotlin compile-level check | PASS (partial) | compiled `platforms/android/app` + `core-bridge` with Gradle/Kotlin against the Android 14 framework jar from Maven Central (`org.robolectric:android-all:14-robolectric-10818077`) with a signature stub for `androidx.core.app.NotificationCompat` (androidx is served from dl.google.com). First compile ever found one real error (`ramMb` vs `maxRamMb` in `AndroidDevice.kt`), fixed. NOT validated: AGP, resources, manifest merge, APK, linking the JNI `.so` with the NDK, running on ART |
| Kotlin/JVM bridge suite | PASS | 16 tests, both library variants (earlier this session; unchanged) |
| R1 real AI website via the browser teacher | PENDING | site unreachable here and needs your interactive sign-in on a machine with a display; the local mock is NOT evidence for R1 |
| R2/R4 physical phone | PENDING | no device |
| R5 OpenClaw agent turn / real-site tool generation | PENDING | same as R1; an agent turn also needs an OpenClaw model provider |

Steps the operator must perform (cannot be done from this container) are in the section "Your steps" of docs/RUNBOOKS.md (to be added when the first one is needed).

## Session 2026-10-05, part 4 (benchmark rerun, routing and drift work; final state of this session)

Reproduction of the baseline: the full Phase 14 suite was rerun from scratch; all 8 original arms matched the first run exactly (base 185 calls / 41 wrong; serve_uncertain 107 / 47; no_frame_filter 154 / 72; cache_refusals 147 / 41; no_router_update 292 / 28; no_monitor 184 / 41; feedback_0.1 184 / 41; feedback_1.0 207 / 23). Evidence: `benchmarks/reports/phase14.{json,md}`.

| Item | Result | Evidence |
|---|---|---|
| Probe-based drift detection (CUSUM + capability-isolated probes) vs the window monitor | MEASURED, not enabled | 3 of 9 drifts repaired vs 1 of 9 across 3 seeds, 0 vs 2 spurious repairs, 270 label queries per run; F53 |
| IDF router, teacher synonyms | MEASURED, not enabled | adversarial fabricated KNOWN 0.20 to 0.93 (3 capabilities); 21/36 learned at first request; F54, F55; `benchmarks/reports/routing_study.json`, `routing_phase7_sets.json` |
| Default test suite | PASS | 327 passed, 11 skipped (338 collected) |
| Real-OpenClaw tests (`OPENCLAW_BIN`): browser teacher/generator against the mock, skill loading, contract | PASS (mock site and real OpenClaw 2026.6.35 only) | `OPENCLAW_BIN=... pytest tests/test_browser_teacher.py tests/test_phase13.py tests/test_phase8.py -k real`: 10 passed in 556 s |
| R1, R2, R3 (live search), R4, R5 | STATUS UNCHANGED: BLOCKED (network policy) or PENDING (needs your PC/phone/sign-in) | parts 1-3 above |

What you need to do yourself (exact steps, one at a time, will be given when you are ready to run them on your own machines): (1) on a PC with a display, install OpenClaw and its gateway, start `openclaw browser --browser-profile openclaw start` (not headless), open the chosen AI website and sign in yourself; (2) tell me the site so the `SiteAdapter` can be tuned and the site's terms checked by you; (3) build and install the Android app on your phone with the SDK/NDK on your PC (`scripts/build_android_libs.sh`, `gradle :app:assembleDebug`, `adb install`). None of these were executed here.
