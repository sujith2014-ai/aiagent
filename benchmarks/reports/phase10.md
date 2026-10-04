# Phase 10 summary (2026-10-04, Linux-6.18.44-fc-v64-x86_64-with-glibc2.39)

## Android target builds (compile-level only: static libraries, nothing linked or run on Android)

Toolchain stand-in for the full variant: zig 0.16.0 (cc + ar); NOT the Android NDK. The lite variant needs no C toolchain.

| target | variant / profile | built | static library | build seconds |
|---|---|---|---|---|
| aarch64-linux-android | lite (mlp-lite backend only, NO C toolchain) / release-small | True | 11.4 MB | 0.2 |
| aarch64-linux-android | full (adds tract; needs a C/asm toolchain, zig stand-in) / release | True | 63.4 MB | 119.2 |
| aarch64-linux-android | full (adds tract; needs a C/asm toolchain, zig stand-in) / release-small | True | 34.8 MB | 159.0 |
| x86_64-linux-android | lite (mlp-lite backend only, NO C toolchain) / release-small | True | 10.9 MB | 0.1 |
| x86_64-linux-android | full (adds tract; needs a C/asm toolchain, zig stand-in) / release | True | 65.8 MB | 117.8 |
| x86_64-linux-android | full (adds tract; needs a C/asm toolchain, zig stand-in) / release-small | True | 29.5 MB | 152.5 |
| armv7-linux-androideabi | lite (mlp-lite backend only, NO C toolchain) / release-small | True | 8.8 MB | 0.2 |
| armv7-linux-androideabi | full (adds tract; needs a C/asm toolchain, zig stand-in) / release | False | - | 0.9 |
| armv7-linux-androideabi | full (adds tract; needs a C/asm toolchain, zig stand-in) / release-small | False | - | 9.5 |

Note: `build seconds` for the lite rows reflect an incremental rebuild (cache hit). From-scratch lite builds measured earlier took 20-29 s per target; full builds take 2-3 minutes.

Host (x86_64) stripped size-optimised JNI library: {"full (with tract)": "9.1 MB", "lite (mlp-lite only)": "1.3 MB"}. A linked Android .so is expected to be of similar size but was not produced (no NDK linker).

## Kotlin/JVM bridge tests (real Rust core through the real JNI library): 14/14 passed

- ok: guardedDeviceToolsAskThePolicyBeforeTouchingTheDevice
- ok: serverToDeviceDeliveryInstallsGoodPackagesAndRejectsBadOnesAndCleansUp
- ok: planExecutionThroughTheBridgeComposesInstalledModules
- ok: misuseNeverCrashesTheProcess
- ok: packagesBuiltOnPcInstallThroughTheFullActivationSequence
- ok: manyThreadsShareOneRuntimeSafely
- ok: hostileAndIncompatiblePackagesAreRejectedAndLeaveNothingBehind
- ok: nativeLibraryReportsCoreVersionAndBackend
- ok: policyChecksRunInTheCoreAndFailClosedOnBadInput
- ok: registrySurvivesCloseAndReopenAndAFreshInstallReproducesTheSameOutputs
- ok: solveMatchesTheRustCliOutputsForTheSamePackages
- ok: unknownTasksAndOutOfRangeInputsAreRefusedNotFabricated
- ok: deviceSpecificCapabilityIsAcceptedOnlyWhereTheDeviceProvidesIt
- ok: nonFiniteInputsAreRefusedNeverAnsweredWithGarbage

## Pending physical/SDK validation

- load libaicore_jni.so on ART/bionic
- run on an arm64 phone: latency, memory, battery, temperature
- Gradle/AGP build of the app module and APK size
- permissions flow, notification/clipboard tools, lifecycle/process death
- NDK-linked .so (only static libraries were produced here)
- 16 KB page-size alignment on newer Android versions