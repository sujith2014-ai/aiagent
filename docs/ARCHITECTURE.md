# Architecture (current state)

```
core/ (Rust crate `aicore`, platform-neutral)
  capability.rs  manifest types        package.rs   .cap read + signature/hash/compat checks
  trust.rs       Ed25519 trust roots   registry.rs  persistent capability registry (versions, stats, rollback)
  router.rs      keyword router (Router trait for learned routers later)
  workspace.rs   per-task working state    trace.rs   JSONL trace of imports and tasks
  backend.rs     Backend/Model traits + ONNX(tract)   device.rs  DeviceProfile (the only platform seam)
  graph.rs       execution plans (cap/select/cond_swap/affine/gather/if/repeat), static validation, step budget
  runtime.rs     guarded import, lazy load/unload (LRU), solve, run_plan, regression
platforms/desktop/  `aicli` harness (arg parsing, JSON, peak RSS); stateless commands: policy-check, approve, features
apps/desktop/       desktop service + REPL (token-protected loopback JSON API, teach-by-example), see ADR-025
training/ integrations/teacher/ packages/  Python research side (never imported by the core)
```
Directory deviations from the master layout: core modules are flat files in `core/src` instead of subdirectories (`core/router`, etc. are placeholders); `packages/capbuild.py` holds signing/building. Revisit when modules grow.

## Phases
0 spec; 1 Rust core + registry + .cap + signing + ONNX backend; 2 training pipeline, LearningPackage, capability A; 3 capabilities B/C, sequential learning, regression, fresh-runtime test, constrained simulation; 4 dynamic routing/composition; 5 novelty detection + TeacherProvider; 6 real teacher + selective learning; 7 consolidation + learned routing + baselines; 8 OpenClaw; 9 desktop app; 10 Android; 11 hybrid execution + delivery + offline; 12 on-device learning; 13 tool growth; 14 long-running benchmark; 15 final evaluation. Status of each is tracked in docs/STATUS.md.
