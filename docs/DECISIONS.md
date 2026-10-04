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
