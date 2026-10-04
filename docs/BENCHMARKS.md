# Benchmarks
Format `bench/1`: one JSON per experiment in `benchmarks/reports/`, plus an auto-generated markdown summary (numbers only from the JSON). Every phase gate re-runs its experiment and re-generates both.
Tracked now: per-capability accuracy on unseen examples, forgetting (drop vs first measurement), bundled regression accuracy, teacher calls, related-encounter local resolution, active/total parameters, package bytes, per-inference latency (p50/p95), peak RSS, constrained-vs-full output differences, conventional baselines (joint and sequential shared MLP).
Phase 4 adds: composite accuracy vs end-to-end baselines, routing/glue overhead (intent vs id), modular-vs-monolithic latency. Not yet tracked: OpenClaw dependency, energy/battery, CPU/GPU utilisation.
See `benchmarks/reports/phase3.md` for the latest numbers.
