# Learning pipeline (Phases 2-3)

unknown task (`NEEDS_HELP` from the runtime) -> `HelpRequest` (minimum context only) -> `TeacherProvider.respond` -> `LearningPackage` (`training/learning_package/schema.py`) -> `validate` -> `BuildService.build` (isolated training of a *new* module, held-out generalization test, ONNX export + PyTorch parity check, signed `.cap`) -> runtime `import` (full activation sequence) -> retry.

- Teacher output is never trained on directly: it must parse into a valid LearningPackage (schema, label ranges, no train/validation leakage, provenance, information-kind rules: MEMORY/KNOWLEDGE never trigger neural training).
- Candidate gate: held-out accuracy >= 0.95 (two architecture attempts allowed), then bundled tests re-run by the runtime on the packaged ONNX.
- Strategy used so far: `new_module` only. Other strategies in the schema (adapter, adapt_existing, ...) are not implemented yet (Phase 6).
- Existing modules are separate frozen artifacts; learning a new capability cannot change them. **This makes "no forgetting" true by construction** for independent modules; it says nothing yet about shared representations, adapters or composition.
- The teacher here is a simulator that owns the ground-truth generators; the build service also uses it as a held-out oracle. Real teachers (Phase 6) will need independent verification.

## Phase 6 additions
- Teacher actions: `new_capability` (LearningPackage), `new_capability_spec` (TaskSpec + environment verification), `reroute`, `use_memory`, `external_research`, `request_tool`, `cannot_help`.
- Strategy engine (`training/strategies.py`, `SelectiveLearner`) and `BuildService.repackage/adapt/commit`; Escalator performs `router_update` for verified reroutes. See DECISIONS ADR-013..015 and FINDINGS F10-F13.
- Real providers: `integrations/teacher/http_providers.py` (OpenAI-compatible, Anthropic). Live validation is PENDING (docs/RUNBOOKS.md R1).
