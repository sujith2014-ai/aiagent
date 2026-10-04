"""Compile-level portability check: builds the whole core + JNI bridge as static libraries for Android targets.
No NDK is available here, so zig (pip install ziglang) stands in as the C/assembler toolchain for tract's ARM kernels; nothing is linked or run.
Writes benchmarks/reports/phase10_targets.json."""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
zig = subprocess.run([sys.executable, "-m", "ziglang", "version"], capture_output=True, text=True).stdout.strip()
d = Path(tempfile.mkdtemp())
(d / "ar").write_text(f'#!/bin/sh\nexec {sys.executable} -m ziglang ar "$@"\n')
specs = {"aarch64-linux-android": ("aarch64-linux-android", ""), "x86_64-linux-android": ("x86_64-linux-android", ""), "armv7-linux-androideabi": ("arm-linux-androideabi", "-mcpu=cortex_a9")}
res = {"toolchain_standin": f"zig {zig} (cc + ar); NOT the Android NDK", "targets": {}}
for triple, (ztarget, extra) in specs.items():
    cc = d / f"cc-{triple}"
    cc.write_text(f'#!/bin/bash\nargs=()\nfor a in "$@"; do case "$a" in --target=*) ;; *) args+=("$a");; esac; done\nexec {sys.executable} -m ziglang cc -target {ztarget} {extra} "${{args[@]}}"\n')
    os.chmod(cc, 0o755); os.chmod(d / "ar", 0o755)
    env = {**os.environ, f"CC_{triple.replace('-', '_')}": str(cc), f"AR_{triple.replace('-', '_')}": str(d / "ar")}
    entry = {}
    for variant, flags, venv in (("lite (mlp-lite backend only, NO C toolchain)", ["--no-default-features"], {k: v for k, v in os.environ.items() if not k.startswith(("CC_", "AR_"))}),
                                 ("full (adds tract; needs a C/asm toolchain, zig stand-in)", [], env)):
        for profile in ("release-small",) if flags else ("release", "release-small"):
            t0 = time.time()
            p = subprocess.run(["cargo", "rustc", "-p", "aicore-jni", f"--profile={profile}", "--target", triple, "--lib", "--crate-type", "staticlib", *flags], cwd=ROOT, env=venv, capture_output=True, text=True)
            lib = ROOT / "target" / triple / profile / "libaicore_jni.a"
            err = [l for l in p.stderr.splitlines() if "error" in l.lower()][:3]
            entry[f"{variant} / {profile}"] = {"ok": p.returncode == 0, "seconds": round(time.time() - t0, 1), "static_lib_bytes": lib.stat().st_size if p.returncode == 0 and lib.exists() else None, "first_errors": err if p.returncode else []}
    res["targets"][triple] = entry
subprocess.run(["cargo", "build", "--profile", "release-small", "-p", "aicore-jni", "--no-default-features", "--target-dir", "target/lite"], cwd=ROOT, capture_output=True)
subprocess.run(["cargo", "build", "--profile", "release-small", "-p", "aicore-jni"], cwd=ROOT, capture_output=True)
sz = lambda p: (ROOT / p).stat().st_size if (ROOT / p).exists() else None
res["host_stripped_size_optimised_so_bytes"] = {"full (with tract)": sz("target/release-small/libaicore_jni.so"), "lite (mlp-lite only)": sz("target/lite/release-small/libaicore_jni.so")}
(ROOT / "benchmarks/reports/phase10_targets.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)[:1800])
