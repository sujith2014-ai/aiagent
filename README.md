# aiagent: self-growing modular AI (research prototype)

Question: can a small runtime progressively acquire reusable neural capabilities, compose them, and reduce dependence on a large external teacher model?
Read `DESIGN.md`, `docs/ARCHITECTURE.md`, `docs/STATUS.md`, and the latest results in `benchmarks/reports/phase3.md`.

Quick start: `cargo build --release && python3 -m pytest tests -q && python3 scripts/run_phase3.py && python3 scripts/summarize.py`

Desktop service: `python3 -m apps.desktop.app init --dir state && python3 -m apps.desktop.app serve --config state/desktop.json` (see docs/ARCHITECTURE.md, FINDINGS F26).
