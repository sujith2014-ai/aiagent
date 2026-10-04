# Android (Phase 10)

**Status: bridge and core portability validated on the JVM and at compile level; execution on Android (ART/bionic) and on physical hardware is PENDING.**

## Architecture
```
Kotlin shell (app module: UI, permissions, lifecycle, storage paths, DeviceTools)   <- NOT compiled here (no Android SDK reachable)
   |  reuses, unchanged:
core-bridge (pure Kotlin/JVM): NativeCore (JNI declarations), CoreClient, GuardedDeviceTools, CapInstaller   <- built and tested with Gradle
   |  JNI: every call returns a JSON string, `{"error": ...}` on failure; panics never cross the boundary
aicore-jni (Rust cdylib/staticlib)  ->  aicore (platform-neutral core: registry, validation, policy, routing, plans, backends)
```
Android supplies only: a `DeviceProfile` (what the phone has), `DeviceTool` implementations (notifications, clipboard; camera/microphone/files deliberately not requested), trust roots (public keys in assets), and storage paths. Packages are the same `.cap` files the desktop produces: **no format or capability-semantics change was needed** (a change would be a portability failure).

## Validated here
- The whole core and JNI bridge compile for `aarch64-linux-android`, `armv7-linux-androideabi` and `x86_64-linux-android` as static libraries (benchmarks/reports/phase10.md). The **lite** variant (mlp-lite backend only) needs no C toolchain at all; the **full** variant (adds tract) needs a C/assembler toolchain for its ARM kernels (zig stood in for the NDK: arm64 and x86_64 compile, armv7 does not).
- A Kotlin/JVM Gradle module drives the real Rust core through the real JNI library (host build of the same crate) with 14 tests: install through the full activation sequence, outputs equal to the Rust CLI's for 60 independently computed cases (1e-6), refusals (unknown task, out-of-range, non-finite, wrong shape), tampered/untrusted/incompatible/garbage packages rejected with nothing left behind, device-specific capability accepted only where the profile provides it, plan execution, policy checks and fail-closed behaviour, device tools gated by policy, misuse (closed/unknown handles, bad arguments, huge input) without crashes, 8 threads on one runtime, persistence across reopen, fresh install, server-to-device delivery (size cap, SHA-256 pin, no leftovers). The suite passes for both library variants.
- Library size: stripped, size-optimised host build 9.1 MB (full) vs 1.3 MB (lite).

## Not validated (PENDING; runbook R2)
Loading the `.so` on ART/bionic; any run on a phone (latency, memory, battery, temperature); the Gradle/AGP build of `app/` and APK size; permission flows, notification/clipboard tools, lifecycle and process death; an NDK-linked `.so` (only static libraries were produced); 16 KB page-size alignment; armv7 execution; on-device learning (Phase 12). The Kotlin files in `app/` were written carefully but have never been compiled.

## Build (needs the Android SDK + NDK)
`scripts/build_android_libs.sh` (lite variant) then `cd platforms/android && gradle :app:assembleDebug`. See docs/RUNBOOKS.md R2.
