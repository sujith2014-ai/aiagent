"""mlp-lite vs tract on the real-data modules exported by Phase 9 (iris 4-d, wine 13-d, tumor 30-d, digits 64-d): per-inference latency, wall time per task, peak RSS.
Requires `python3 scripts/run_phase9.py` to have produced runs/phase9/export and the test row files."""
import json, shutil, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.cli import Cli

RUN = ROOT / "runs" / "phase9"
trust = RUN / "state" / "trust.json"
out = {"benchmark_format": "bench/1", "experiment": "backend-bench", "date": time.strftime("%Y-%m-%d"), "rows_per_task": None, "tasks": {}}
work = ROOT / "runs" / "backend_bench"; shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
intents = {"iris_species": "classify iris flower species from sepal and petal measurements", "wine_cultivar": "identify wine cultivar from chemical analysis",
           "digit_recognition": "recognize handwritten digit from 8x8 pixel intensities", "tumor_malignancy": "classify breast tumor as malignant or benign from cell measurements"}
for backend in ("mlp", "tract"):
    cli = Cli(work / f"rt_{backend}", trust)
    for f in sorted((RUN / "export").glob("*.cap")):
        rep = cli._run_detect("full", "import", str(f), backend=backend)[0]; assert rep["activated"], rep
    for cap in intents:
        rows = [json.loads(l) for l in (RUN / f"test_{cap}.jsonl").read_text().splitlines() if l.strip()]
        rows = (rows * (6000 // len(rows) + 1))[:6000]
        cf = work / f"{cap}_{backend}.jsonl"; cf.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        s = cli._run_detect("full", "batch", "--cases", str(cf), backend=backend)["summary"]
        out["tasks"].setdefault(cap, {})[backend] = {"latency_us_p50": s["latency_us_p50"], "latency_us_p95": s["latency_us_p95"], "wall_us_per_task": round(1000 * s["wall_ms"] / s["cases"], 2), "peak_rss_kb": s["peak_rss_kb"], "cases": s["cases"]}
out["rows_per_task"] = 6000
(ROOT / "benchmarks/reports/backend_bench.json").write_text(json.dumps(out, indent=1))
for cap, v in out["tasks"].items():
    print(f"{cap:18s} mlp-lite p50 {v['mlp']['latency_us_p50']}us wall {v['mlp']['wall_us_per_task']}us rss {v['mlp']['peak_rss_kb']}KB | tract p50 {v['tract']['latency_us_p50']}us wall {v['tract']['wall_us_per_task']}us rss {v['tract']['peak_rss_kb']}KB")
