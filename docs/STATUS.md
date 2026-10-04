# Phase status
| Phase | Status | Evidence |
|---|---|---|
| 0 Spec/ADRs/benchmark format | done | DESIGN.md, DECISIONS.md, BENCHMARKS.md |
| 1 Core skeleton, registry, .cap, signing, ONNX | done | cargo tests, tests/test_acceptance.py |
| 2 Training pipeline, LearningPackage, capability A | done | tests/test_learning_package.py, phase3 report |
| 3 B/C, sequential learning, regression, restart, constrained sim | done (simulation only) | benchmarks/reports/phase3.* |
| 4 Dynamic routing, reuse, composition, execution graph | done (hand-written plans; see FINDINGS F2) | benchmarks/reports/phase4.*, tests/test_graph.py |
| 5-15 | not started | |
