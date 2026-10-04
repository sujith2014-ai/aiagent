# Architecture Decision Records

## ADR-001: Rust core, Python for training
Status: accepted. One portable implementation for Windows/Linux/macOS/Android; Python/PyTorch only for research and training.
Consequence: the core crate (`core/`) contains no OS-specific code; platforms provide `DeviceProfile` + `Backend`. Python must not hold runtime business logic (only the `scripts/cli.py` wrapper).

## ADR-002: PyTorch for training
Status: accepted. Fast architecture iteration. Risk noted: the legacy TorchScript ONNX exporter (`dynamo=False`) used in `training/exporters/onnx_export.py` is deprecated; the exporter is isolated in one file so it can be replaced (or the graph built by hand).

## ADR-003: ONNX first, format declared per package
Status: accepted. `model.format` / `model.variant` are manifest fields; runtime refuses formats without a backend (`compatibility` step). The ONNX backend is `tract` (pure Rust), not `ort`, so no native binary is needed on Android/CI. Revisit if tract lacks operators/performance for larger modules (open question for Phase 10).

## ADR-004: Kotlin shell + Rust core via JNI (Android)
Status: accepted, not yet implemented (Phase 10). Android may only supply a `DeviceProfile`, `DeviceTools` and a `Backend`; any change to package format or capability semantics for Android is a portability failure.

## ADR-005: `.cap` package format `cap/1`
Status: accepted. Zip containing `manifest.json`, `signature.json`, `model/`, `tests/`, `routing/`. See CAPABILITY_FORMAT.md. Signature is over the *exact stored bytes* of `manifest.json` (no canonicalization rules to disagree on across languages).

## ADR-006: Ed25519 trust model
Status: accepted. Explicit trust-root file (`key_id -> public key`). Signature validity is necessary but never sufficient for activation. The teacher side has no access to signing code/keys (enforced by `tests/test_acceptance.py::test_teacher_side_has_no_signing_access`); a separate build/validation service trains, evaluates and signs.

## ADR-007: OpenClaw behind an integration boundary
Status: accepted, not yet implemented (Phase 8). OpenClaw is used for external actions (web, browser, files, shell, APIs); no deep fork; extensions/plugins preferred. The core only sees an `ActionProvider`-style interface plus the policy layer.

## ADR-008: Keyword router as the baseline router
Status: accepted for Phases 1-3. Deterministic; scores registry metadata (intent-token coverage of declared keywords, input shape). It is deliberately simple so a learned router (Phase 7) has a real baseline. Known weakness: paraphrases with no keyword overlap yield NEEDS_HELP.

## ADR-009: tract-based "sandbox"
Status: accepted with a stated limitation. "Sandbox load" in Phase 1 means: model parsed by a memory-safe pure-Rust backend after signature/hash/compat checks, with declared-vs-actual size verification and a probe inference. It is NOT OS-level isolation. See SECURITY.md.

## ADR-010: Additive optional manifest fields stay within `cap/1`
Status: accepted. `input_stats` (training input min/max/mean/std) and `calibration` (temperature) were added as optional fields: old packages still load (tested by the earlier acceptance tests, which build packages without them); runtimes ignoring them still work. Because they are covered by the manifest signature they cannot be stripped or altered in transit. A change that older runtimes cannot ignore would require `cap/2`.

## ADR-011: Unknown handling policy
Status: accepted. Out-of-domain input (outside training range + 10% margin on any feature) returns NEEDS_HELP(OUT_OF_DISTRIBUTION); low calibrated confidence, ambiguous routing, or poor history downgrade to UNCERTAIN (answer returned but flagged); no match returns NEEDS_HELP. Uncertain answers do not trigger the teacher automatically. Signals are individually switchable (`--detect keyword|novelty|confidence|calibrated|full`) so they can be ablated.

## ADR-012: Teacher output is untrusted text
Status: accepted. `TeacherProvider.advise` returns raw JSON text; `parse_response` validates it strictly (known action, required fields, referenced capabilities must be installed); LearningPackages are validated again before any training. One teacher call per task maximum; offline requests are queued; research/tool actions are reported as NEEDS_EXTERNAL until OpenClaw exists.

## ADR-013: Teachers return declarative specs, not data or code
Status: accepted. Real LLMs cannot reliably produce large labelled datasets and generated code must not run in the learning pipeline. The teacher returns a `TaskSpec` (domain, restricted expression, worked examples); labels come from a whitelist AST evaluator (`training/safe_expr.py`, no eval/exec, node/magnitude limits). A direct-LearningPackage action remains for providers that can supply data.

## ADR-014: Environment verification is the promotion authority
Status: accepted. Teacher output is never ground truth. A spec-derived candidate is promoted only if it also reaches >= 0.95 on labelled examples supplied by the environment/user, independent of the teacher; without such examples a spec cannot be installed (NEEDS_HELP). Known limit: errors below the gate resolution are not detectable (FINDINGS F10).

## ADR-015: Selective learning order and gates
Status: accepted. Cheapest first: metadata_update (no neural change) -> router_update (re-sign with new keywords) -> adapt_existing with replay -> new_module. Gates: new-domain verification >= 0.95; old-domain regression drop <= 1 point; routing regression for router/metadata changes; failed candidates are never installed, installed versions can be rolled back. The remaining strategies in the schema (adapter, new_connection, module_expansion) are not implemented.

## ADR-016: Archive instead of delete; routers and capabilities share one package format
Status: accepted. Archiving hides a capability from routing but keeps its packages in the store (restorable); it is refused while an active capability declares a dependency on it. A learned router is a normal signed `.cap` with `role: "router"` (optional additive manifest field): it goes through the same activation sequence, is excluded from capability listings and is never routed to. Nothing is ever permanently deleted by consolidation.

## ADR-017: Keyword router stays the default; learned router is optional and not scaled
Status: accepted. Evidence in FINDINGS F16: the learned router improves paraphrase recall but is not clearly safer on adversarial intents, is >10x larger than the modules, adds ~13 us per task and must be retrained for every new capability (stale routers cannot reach new capabilities). `--router learned` is available for experiments; defaults and the escalation loop use the keyword router.

## ADR-018: Consolidation gates
Status: accepted. Merge: keeper must pass the duplicate's bundled tests, inherit its keywords, and all previously served routes must still resolve KNOWN to the same (or merged) capability with the duplicate archived; otherwise restore + rollback. Compaction (distillation/pruning): held-out accuracy within 1 point of the original, bundled tests pass, version kept for rollback. Admin/probe traffic never counts as usage. Duplicate detection compares only capabilities with equal input dimension and label set, on in-distribution probes.

## ADR-019: OpenClaw is used through its narrow `infer` CLI, as a PC/server component
Status: accepted. `openclaw infer web search|fetch` (and optionally `infer model run`) via an isolated subprocess (own profile and HOME, minimal environment, output size cap, timeout). No `openclaw agent` turns, no OpenClaw shell/browser/file/channel tools, no credentials or internal state shared. The ~389 MB Node package cannot run on Android; mobile devices delegate research to a PC/server.

## ADR-020: Deterministic default-deny policy in the Rust core; operator-only approvals
Status: accepted. Every external action is evaluated by `core/src/policy.rs` before any provider runs. First matching rule wins; a matched rule whose constraints fail denies (no fall-through). Hard guards: embedded URL credentials, non-http(s), loopback/private/link-local/metadata hosts (unless the operator sets `allow_private_hosts`), secret-looking text, per-task limits. Approvals are bound to the exact request hash and expire; they can only be created through the operator CLI command `approve`, which no AI-facing code references (tested). Engine errors fail closed. Every decision is audit-logged.

## ADR-021: External evidence is untrusted data
Status: accepted. Retrieved text is passed to the teacher as delimited data with an explicit instruction not to follow it; it is stored with hash and provenance; learning from it still requires environment verification (ADR-014); one research round per task; simulated evidence is flagged `simulated: true` end to end.

## ADR-022: Pin the OpenClaw version used for validation
Status: accepted. Version drift (npm latest vs tarball, Node requirements) is a known risk; contract tests run against a pinned install via `OPENCLAW_BIN`; the adapter parses results defensively and maps every failure to a typed `ActionUnavailable`.
