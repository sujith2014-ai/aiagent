# Benchmarks
Format `bench/1`: one JSON per experiment in `benchmarks/reports/`, plus an auto-generated markdown summary (numbers only from the JSON). Every phase gate re-runs its experiment and re-generates both.
Tracked now: per-capability accuracy on unseen examples, forgetting (drop vs first measurement), bundled regression accuracy, teacher calls, related-encounter local resolution, active/total parameters, package bytes, per-inference latency (p50/p95), peak RSS, constrained-vs-full output differences, conventional baselines (joint and sequential shared MLP).
Phase 4 adds: composite accuracy vs end-to-end baselines, routing/glue overhead (intent vs id), modular-vs-monolithic latency. Not yet tracked: OpenClaw dependency, energy/battery, CPU/GPU utilisation.
See `benchmarks/reports/phase3.md` for the latest numbers.

Phase 5-7 reports: `phase5.*` (detection ablation, escalation), `phase6.*` (fault matrix, selective learning), `phase7.*` (consolidation, routing comparison, cross-phase baseline table). Small-quantized-LLM baseline: PENDING (no weights/egress here).

Phase 10: `phase10.*` (Android target builds, Kotlin/JVM bridge results), `phase10_targets.json`, `backend_bench.json` (mlp-lite vs tract latency/memory on the four real-data modules). All earlier reports were regenerated under the mlp-lite default backend (accuracy unchanged, latency lower).
