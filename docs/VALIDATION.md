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
