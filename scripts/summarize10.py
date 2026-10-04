"""Assemble benchmarks/reports/phase10.json/.md from the Android target build record and the Kotlin/JVM bridge test results."""
import glob, json, time, platform, xml.etree.ElementTree as ET
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
tg = json.loads((ROOT / "benchmarks/reports/phase10_targets.json").read_text())
xmls = glob.glob(str(ROOT / "platforms/android/core-bridge/build/test-results/test/*.xml"))
kt = []
for f in xmls:
    r = ET.parse(f).getroot()
    kt = [{"test": tc.get("name").rstrip("()"), "ok": tc.find("failure") is None and tc.find("error") is None} for tc in r.findall("testcase")]
rep = {"benchmark_format": "bench/1", "experiment": "phase10-android-portability", "platform": platform.platform(), "date": time.strftime("%Y-%m-%d"),
       "targets": tg, "kotlin_jvm_bridge_tests": kt,
       "pending_physical_validation": ["load libaicore_jni.so on ART/bionic", "run on an arm64 phone: latency, memory, battery, temperature", "Gradle/AGP build of the app module and APK size", "permissions flow, notification/clipboard tools, lifecycle/process death",
                                       "NDK-linked .so (only static libraries were produced here)", "16 KB page-size alignment on newer Android versions"]}
(ROOT / "benchmarks/reports/phase10.json").write_text(json.dumps(rep, indent=1))
mb = lambda b: "-" if b is None else f"{b / 1e6:.1f} MB"
L = [f"# Phase 10 summary ({rep['date']}, {rep['platform']})", "", "## Android target builds (compile-level only: static libraries, nothing linked or run on Android)", "", f"Toolchain stand-in for the full variant: {tg['toolchain_standin']}. The lite variant needs no C toolchain.", "",
     "| target | variant / profile | built | static library | build seconds |", "|---|---|---|---|---|"]
for t, e in tg["targets"].items():
    for k, v in e.items():
        L.append(f"| {t} | {k} | {v['ok']} | {mb(v['static_lib_bytes'])} | {v['seconds']} |")
L += ["", "Note: `build seconds` for the lite rows reflect an incremental rebuild (cache hit). From-scratch lite builds measured earlier took 20-29 s per target; full builds take 2-3 minutes.", "", f"Host (x86_64) stripped size-optimised JNI library: {json.dumps({k: mb(v) for k, v in tg['host_stripped_size_optimised_so_bytes'].items()})}. A linked Android .so is expected to be of similar size but was not produced (no NDK linker).", "",
      f"## Kotlin/JVM bridge tests (real Rust core through the real JNI library): {sum(t['ok'] for t in kt)}/{len(kt)} passed", ""] + [f"- {'ok' if t['ok'] else 'FAIL'}: {t['test']}" for t in kt]
L += ["", "## Pending physical/SDK validation", ""] + [f"- {x}" for x in rep["pending_physical_validation"]]
(ROOT / "benchmarks/reports/phase10.md").write_text("\n".join(L)); print("\n".join(L[:30]))
