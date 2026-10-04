# Phase status
| Phase | Status | Evidence |
|---|---|---|
| 0 Spec/ADRs/benchmark format | done | DESIGN.md, DECISIONS.md, BENCHMARKS.md |
| 1 Core skeleton, registry, .cap, signing, ONNX | done | cargo tests, tests/test_acceptance.py |
| 2 Training pipeline, LearningPackage, capability A | done | tests/test_learning_package.py, phase3 report |
| 3 B/C, sequential learning, regression, restart, constrained sim | done (simulation only) | benchmarks/reports/phase3.* |
| 4 Dynamic routing, reuse, composition, execution graph | done (hand-written plans; see FINDINGS F2) | benchmarks/reports/phase4.*, tests/test_graph.py |
| 5 Unknown/novelty detection, NEEDS_HELP flow, TeacherProvider, simulator | done, with documented failures (FINDINGS F6-F9) | benchmarks/reports/phase5.*, tests/test_phase5.py |
| 6 Real-teacher integration, selective learning, automated candidate evaluation | done locally; live teacher PENDING (R1) | benchmarks/reports/phase6.*, tests/test_phase6.py |
| 7 Consolidation/pruning, learned routing, baseline comparison | done; small-LLM baseline PENDING (no weights/egress) | benchmarks/reports/phase7.*, tests/test_phase7.py |
| 8 OpenClaw integration, research/tool escalation, provenance, retry-after-learning | boundary + policy + adapter done and contract-validated against the real binary; **live research PENDING (R3)**; research results use a simulated provider/teacher | benchmarks/reports/phase8.*, tests/test_phase8.py, docs/OPENCLAW.md |
| 9 Desktop runtime/application, real-world controlled tasks | done (4 small public datasets; on par with standard baselines, not better; novelty-rule redesign, FINDINGS F23-F27) | benchmarks/reports/phase9.*, novelty_rules.json, tests/test_phase9.py |
| 10 Android Kotlin shell, Rust JNI, portable inference, .cap installation | core + JNI compile for Android targets; Kotlin/JVM bridge validated (14 tests, both library variants); **app module uncompiled and all on-device execution PENDING (R2)**; mlp-lite backend added | benchmarks/reports/phase10.*, backend_bench.json, tests/test_phase10.py, core/tests/mlp_lite.rs, docs/ANDROID.md |
| 11 Hybrid local/server execution, signed capability delivery, offline mode | done on loopback (no real network, no TLS); delivery attacks defended 6/6; single signing key for catalog and packages is a stated weakness | benchmarks/reports/phase11.*, tests/test_phase11.py, apps/server, apps/hybrid |
| 12 On-device learning | done on PC (Rust trainer, device-key signing, adapt/alias gates, server endorsement, JNI/Kotlin bindings tested on the JVM); accuracy at parity with server training, adaptation gated and sometimes refused (F36-F39); **phone battery/thermal/NPU PENDING (R4); Android Keystore not implemented** | benchmarks/reports/phase12.*, tests/test_phase12.py |
| 13 OpenClaw tools/skills, controlled tool growth | done locally: gates, sandbox, operator-bound approval, signed install, skill export validated against real OpenClaw; **generator is scripted (live model PENDING, R5)**; CPython sandbox is research-grade (F41) | benchmarks/reports/phase13.*, tests/test_phase13.py |
| 14-15 | not started | |
