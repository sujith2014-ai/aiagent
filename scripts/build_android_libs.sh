#!/bin/bash
# Builds libaicore_jni.so for Android ABIs and places them where the app module expects them.
# REQUIRES the Android NDK for linking (not available in the research container: dl.google.com was unreachable). Not run in CI. See docs/RUNBOOKS.md R2.
# Builds the LITE variant (mlp-lite backend only): it compiles for every Android target with plain rustc, no C toolchain needed to compile; the NDK linker is only used to link the .so.
# Add `--features tract` instead of --no-default-features only if you need the general ONNX runtime (that needs the NDK clang for its ARM assembly kernels).
set -euo pipefail
: "${ANDROID_NDK_HOME:?set ANDROID_NDK_HOME to an NDK r26+ install}"
API=26
TOOLCHAIN="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/$(uname -s | tr A-Z a-z)-x86_64/bin"
cd "$(dirname "$0")/.."
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android
for spec in "aarch64-linux-android:arm64-v8a:aarch64-linux-android${API}-clang" "armv7-linux-androideabi:armeabi-v7a:armv7a-linux-androideabi${API}-clang" "x86_64-linux-android:x86_64:x86_64-linux-android${API}-clang"; do
  IFS=: read -r triple abi clang <<<"$spec"
  upper=$(echo "$triple" | tr a-z- A-Z_)
  export "CARGO_TARGET_${upper}_LINKER=$TOOLCHAIN/$clang" "CC_${triple//-/_}=$TOOLCHAIN/$clang" "AR_${triple//-/_}=$TOOLCHAIN/llvm-ar"
  cargo build --profile release-small -p aicore-jni --no-default-features --target "$triple"
  mkdir -p "platforms/android/app/src/main/jniLibs/$abi"
  cp "target/$triple/release-small/libaicore_jni.so" "platforms/android/app/src/main/jniLibs/$abi/"
done
echo "now: cp <your trust.json> platforms/android/app/src/main/assets/trust.json && (cd platforms/android && gradle :app:assembleDebug)"
