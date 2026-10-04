"""Phase 10 guards: the Rust JNI exports and Kotlin declarations cannot drift apart; the Android manifest stays minimal; the Kotlin/JVM
bridge suite (real core through the real JNI library) can be run for both library variants."""
import json, os, re, subprocess
from pathlib import Path
import pytest
from conftest import ROOT

ANDROID = ROOT / "platforms/android"


def test_every_jni_export_has_a_matching_kotlin_external_and_vice_versa():
    rust = set(re.findall(r"Java_com_aiagent_core_NativeCore_(\w+)", (ANDROID / "jni/src/lib.rs").read_text()))
    kotlin = set(re.findall(r"external fun (\w+)\(", (ANDROID / "core-bridge/src/main/kotlin/com/aiagent/core/NativeCore.kt").read_text()))
    assert rust == kotlin and len(rust) >= 8, (rust ^ kotlin)


def test_jni_exports_never_unwind_or_throw_across_the_boundary():
    src = (ANDROID / "jni/src/lib.rs").read_text()
    exports = [chunk for chunk in src.split("#[no_mangle]")[1:]]
    assert len(exports) >= 8
    for chunk in exports:
        name = re.search(r"fn Java_com_aiagent_core_NativeCore_(\w+)", chunk).group(1)
        body = chunk.split("\n}\n")[0]
        assert "catch_unwind(" in body or name == "nativeVersion", f"{name} can unwind across the JNI boundary"     # nativeVersion only builds a constant document
    assert "throw_new" not in src                                                                                     # failures are JSON, never exceptions


def test_manifest_requests_only_network_and_notifications_and_exposes_one_activity():
    m = (ANDROID / "app/src/main/AndroidManifest.xml").read_text()
    perms = set(re.findall(r'uses-permission android:name="([^"]+)"', m))
    assert perms == {"android.permission.INTERNET", "android.permission.POST_NOTIFICATIONS"}, perms
    assert 'android:allowBackup="false"' in m and 'android:usesCleartextTraffic="false"' in m
    assert m.count('android:exported="true"') == 1
    assert 'cleartextTrafficPermitted="false"' in (ANDROID / "app/src/main/res/xml/network_security_config.xml").read_text()


def test_app_module_reuses_the_validated_bridge_sources_instead_of_forking_them():
    g = (ANDROID / "app/build.gradle.kts").read_text()
    assert '../core-bridge/src/main/kotlin' in g
    assert not list((ANDROID / "app/src").rglob("CoreClient.kt")) and not list((ANDROID / "app/src").rglob("NativeCore.kt"))


def test_signing_keys_are_never_part_of_the_android_tree():
    for f in ANDROID.rglob("*"):
        if f.is_file() and f.suffix in {".kt", ".kts", ".xml", ".json", ".rs", ".toml"}:
            t = f.read_text(errors="ignore")
            assert "private" not in f.name and "BEGIN PRIVATE" not in t and "Ed25519PrivateKey" not in t, f


def test_core_has_no_platform_specific_code():
    banned_cfg = re.compile(r"cfg\s*\(\s*(target_os|target_arch|windows|unix|target_family)")
    for f in (ROOT / "core/src").rglob("*.rs"):
        text = f.read_text()
        assert not banned_cfg.search(text), f"platform-specific cfg in the platform-neutral core: {f}"
        code = "\n".join(l.split("//")[0] for l in text.splitlines())                    # comments may mention platforms; code may not
        assert not re.search(r"\b(android|macos|linux)\b|(?<![.\w])windows\b|\bios\b", code, re.I), f      # `.windows(n)` is the slice method


@pytest.mark.skipif(not os.environ.get("RUN_ANDROID_JVM_TESTS"), reason="set RUN_ANDROID_JVM_TESTS=1 (needs Gradle + Maven Central; ~3 min, builds the JNI library twice)")
@pytest.mark.parametrize("lite", ["0", "1"])
def test_kotlin_bridge_suite_passes_for_both_library_variants(lite):
    import xml.etree.ElementTree as ET
    p = subprocess.run([str(ROOT / "scripts/run_android_jvm_tests.sh")], cwd=ROOT, env={**os.environ, "LITE": lite}, capture_output=True, text=True, timeout=1500)
    res = ET.parse(next((ANDROID / "core-bridge/build/test-results/test").glob("*.xml"))).getroot()
    assert res.get("failures") == "0" and res.get("errors") == "0" and int(res.get("tests")) >= 14, p.stdout[-2000:]


@pytest.mark.skipif(not (ROOT / "benchmarks/reports/phase10_targets.json").exists(), reason="run scripts/android_targets_check.py first")
def test_recorded_android_target_builds_cover_the_lite_variant_for_all_abis():
    r = json.loads((ROOT / "benchmarks/reports/phase10_targets.json").read_text())
    for triple in ("aarch64-linux-android", "armv7-linux-androideabi", "x86_64-linux-android"):
        lite = {k: v for k, v in r["targets"][triple].items() if k.startswith("lite")}
        assert lite and all(v["ok"] for v in lite.values()), triple
