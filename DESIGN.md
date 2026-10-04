# Unified AI System: Design Spec (v0.1)

## 1. Goal
One AI architecture that runs on PC (Windows/Linux/macOS) and Android. Not two AIs.
Learned capabilities belong to the system, not to a device.

## 2. Core components (shared by all runtimes)
- **Capability Registry**: catalogue of capabilities (id, version, inputs/outputs, requirements, tests, module refs).
- **Neural Modules**: learned components in a platform-independent format.
- **Router / Workspace**: selects capabilities for a task, holds working state, uses learned routing info.
- **Runtime layer** (per platform): executes modules via a backend and exposes device tools.
- **Optional server**: OpenClaw (browser/tools/research) + Teacher AI for offloaded training and research.

## 3. Runtimes
| | PC runtime | Android runtime |
|---|---|---|
| Role | Heavy: training, local models, OpenClaw, browser/tools, optional teacher | Light: runs small modules, offloads expensive work |
| Module selection | Full / large variants | Smaller variants (quantized / distilled) |
| Offload | n/a | Training and research go to PC or server |
| Device capabilities | filesystem, desktop, browser | camera, notifications, app control |

## 4. Hard requirement: platform-independent capabilities
A capability must be transferable between devices:
- Learn on PC -> package -> Android downloads and runs.
- Learn small on Android -> sync -> PC can use it.

What must be portable (inside the package): neural module weights, capability metadata, tests, learned routing info.
What may differ: the execution backend (e.g. ONNX Runtime / llama.cpp / PyTorch on PC; ONNX Runtime Mobile / NNAPI / LiteRT on Android) and device-specific capabilities.

### 4.1 Capability package (`.cap`, a zip or directory)
```
capability.json        # manifest
modules/<name>.<variant>.onnx   # weights, one or more size variants
tests/cases.jsonl      # input -> expected/acceptance checks
routing/router_hints.json       # learned routing features/examples
README.md              # optional
```
Manifest (sketch):
```json
{
  "id": "summarize.email",
  "version": "1.2.0",
  "schema": 1,
  "io": {"inputs": [{"name": "text", "type": "string"}], "outputs": [{"name": "summary", "type": "string"}]},
  "modules": [
    {"file": "modules/summ.large.onnx", "variant": "large", "format": "onnx", "min_ram_mb": 4096},
    {"file": "modules/summ.small.onnx", "variant": "small", "format": "onnx", "min_ram_mb": 512}
  ],
  "requires": {"device_caps": []},
  "tests": "tests/cases.jsonl",
  "routing": "routing/router_hints.json",
  "provenance": {"learned_on": "pc", "teacher": null, "created": "..."},
  "hash": "sha256:..."
}
```
Rules:
1. Weights use an interchange format (ONNX first; the format is a manifest field, so others can be added).
2. `requires.device_caps` lists abstract capabilities (e.g. `camera`, `fs.read`, `browser`). The router only offers a capability on a device that provides all of them.
3. A package is accepted on a device only if its bundled tests pass on that device's backend within tolerance.
4. Packages are content-hashed and versioned so sync is a diff of registries.

## 5. Abstractions that keep the core platform-free
- `Backend` interface: `load(module)`, `run(inputs)`, `info()`. One implementation per runtime.
- `DeviceTools` interface: abstract tool names mapped to platform implementations.
- `Registry` sync protocol: list / fetch / push packages by id+version+hash; works over LAN, server, or file copy.
- Core (router, workspace, registry logic) is pure Python with no OS-specific calls; all of it goes through the two interfaces above.

## 6. Phased plan
Superseded by the 16-phase master plan (Phases 0-15) in docs/ARCHITECTURE.md section "Phases". Reconciliation notes:
- v0.1 said "core is pure Python"; **superseded**: the core is Rust (ADR-001). Python is research/training only.
- v0.1 manifest sketch had hashes only; **superseded**: Ed25519-signed manifest with trust roots (ADR-006, docs/CAPABILITY_FORMAT.md).
- v0.1 Phase 1 excluded OpenClaw; **kept**: OpenClaw arrives in Phase 8, after the core learning loop is proven.

## 7. Locked decisions (Phase 1)
1. **First capability:** synthetic relation/comparison classification (two feature vectors + relation request -> LESS/EQUAL/GREATER). The comparison algorithm is never hard-coded in the module. Then learn capabilities B and C sequentially and verify A has not degraded. Loop: unknown task -> capability detector -> teacher simulator -> learning package -> PyTorch training -> candidate -> evaluation -> ONNX export -> signed `.cap` -> registry -> dynamic router -> solve unseen examples without the teacher.
2. **Core language:** Rust. Kotlin is only the Android shell (UI, permissions, lifecycle, `DeviceTools`). Rust owns registry, package validation, routing/workspace, execution. Training stays Python/server-side. No Chaquopy.
3. **Models/training:** PyTorch -> ONNX. The `.cap` spec is independent of both via `model.format` and `model.variant`. Modules are tiny (KB/MB), not LLMs.
4. **Signing:** Ed25519 with explicit trust roots. `.cap` carries content hashes, a signed manifest, signer/key ID, id/version, creation metadata, compatibility requirements. Rejected unless signature, hashes, compatibility and bundled tests all pass. The teacher never holds the production signing key; a separate build/validation service trains, evaluates and signs.

OpenClaw and real teacher models are out of Phase 1 scope.

## 8. Phase 1 hard gates
- One platform-neutral core; no Android/PC business logic in it.
- `.cap` versioned from day one.
- >= 3 separately learned capabilities, learned sequentially; earlier parameters frozen unless adaptation is the experiment.
- Dynamic registry discovery; no `if capability == "comparison"` logic.
- ONNX module unload/reload works.
- Export -> destroy runtime state -> fresh install -> import gives equivalent behavior (A+B+C still work).
- Corrupted package rejected; incompatible package rejected; failed bundled tests block activation.
- Unknown capability returns `NEEDS_HELP`, never a fabricated result.
- Record latency, memory, package size, active parameter count, accuracy.
- Conventional tiny-network baseline for comparison.
- Machine-readable benchmark report after every experiment.
- PC_FULL vs PC_CONSTRAINED (Android-like limits): same `.cap`, same outputs within tolerance.

Phase 1 is an experiment, not an application: no UI, Android app, OpenClaw, database infrastructure or generic agent framework.

Key question: can a tiny runtime acquire several separately trained neural capabilities, dynamically discover and execute them, preserve earlier ones, package them portably, and use them after a clean restart without the teacher?
