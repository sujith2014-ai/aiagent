# Security model and limitations

Implemented (tested in `tests/test_acceptance.py`): Ed25519 signature over the manifest, explicit trust roots, per-file SHA-256, undeclared-file and path-traversal rejection, compatibility checks (format version, runtime version, model format, device capabilities, RAM/model-size limits), dependency check, bundled-test gate, version-must-increase, rollback, re-verification of signature+hashes on every module load from the store, teacher side isolated from signing code.

Not implemented / limitations (do not assume otherwise):
- "Sandbox load" is a memory-safe parser + probe inference, not OS isolation, seccomp or process sandboxing. A malicious ONNX graph could still try to exhaust CPU/memory inside the process (only the declared size and a probe latency are checked).
- No key revocation, rotation or expiry. No transparency log. Trust file is a plain JSON file with no integrity protection.
- Signing key is a file on the build host (`keys/`, gitignored); no HSM.
- No replay/downgrade protection beyond "version must be newer than the active one".
- The policy layer (permissions for external actions), privacy filter for escalation and OpenClaw sandboxing are Phase 5-8 work and do not exist yet.

## Escalation privacy (Phase 5)
Teacher requests carry only: redacted intent (<=200 chars), input dimension, installed capability ids, routing status/candidates, a redacted failure string and tool names. No input values, history, files or credentials. Redaction is regex-based (see FINDINGS F9) and is not a guarantee. Teacher responses are untrusted text (ADR-012).

## Phase 6-7 additions
Teacher rules are evaluated by a whitelist AST interpreter (no eval/exec; node and magnitude limits; 15 hostile expressions rejected in tests) and promoted only after environment verification. Learned routers are signed packages. Consolidation never deletes (archive is reversible), refuses to archive capabilities others depend on, and every compaction/merge is gated and rollback-able. Limits: a teacher rule wrong on a small region passes any tolerance-level gate (FINDINGS F10); routing-regression checks only cover intents seen so far (F11).

## Phase 8: external actions
Policy layer (Rust, default deny, hard guards, approvals bound to request hash, fail closed), action broker with audit log (parameters hashed, redacted previews, no raw secrets), OpenClaw adapter with isolated profile/HOME and minimal environment, evidence as untrusted data with provenance. Limits (FINDINGS F20): string-based host checks (no DNS/redirect resolution), logical-only approval separation inside one OS user, unsigned policy file, heuristic secret detection. Prompt-injection containment is tested with a scripted obedient teacher only, not a real LLM.

## Phase 9: desktop service
Loopback-only JSON API with bearer token, Host and Origin checks, JSON-only input, 5 MB declared-length cap, generic error bodies. The process that serves the API also holds the local signing key (single-user mode), and any holder of the API token can install a verified model. No TLS. Novelty-based refusal is a heuristic with measured false-refusal and miss rates (FINDINGS F24), not a security boundary.

## Phase 10: Android / backends
Package validation is identical on Android (same core); trust roots are public keys in app assets, signing keys never ship. The manifest requests only INTERNET and POST_NOTIFICATIONS, forbids cleartext and backup, and exports one activity (tested statically). The JNI bridge never throws or unwinds across the boundary and returns JSON errors; closed/unknown handles are refused. Model backends are panic-isolated, so a validly signed but malformed model is a rejected import, not a crash; the mlp-lite parser is fuzzed. NaN/Infinity are refused everywhere. Not validated: sandboxing on a real device, ART/bionic behaviour, and anything requiring the Android SDK.
