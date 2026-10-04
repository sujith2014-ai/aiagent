# aiagent: self-growing modular AI (research prototype)

Question: can a small runtime progressively acquire reusable neural capabilities, compose them, and reduce dependence on a large external teacher model?
Read `DESIGN.md`, `docs/ARCHITECTURE.md`, `docs/STATUS.md`, and the latest results in `benchmarks/reports/phase3.md`.

Quick start: `cargo build --release && python3 -m pytest tests -q && python3 scripts/run_phase3.py && python3 scripts/summarize.py`
