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
