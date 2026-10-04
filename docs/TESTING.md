# Testing
- `cargo build --release && cargo test --release` : Rust unit tests (routing tokenizer, softmax, semver, plan validation).
- `python3 -m pytest tests -q` : acceptance tests driving the real Rust binary (35 tests incl. tests/test_graph.py: activation order, corruption, tamper, untrusted signer, incompatibility variants, failed tests, dependencies, version/rollback, NEEDS_HELP, unload/reload, store tampering, teacher isolation, LearningPackage validation).
- `python3 scripts/run_phase4.py && python3 scripts/summarize4.py` : composition experiment (benchmarks/reports/phase4.*).
- `python3 scripts/run_phase3.py && python3 scripts/summarize.py` : the sequential-learning experiment; writes `benchmarks/reports/phase3.json` and `.md`.
Requirements: Rust 1.97+, Python 3.11, `pip install numpy onnx onnxruntime torch cryptography pytest`.
