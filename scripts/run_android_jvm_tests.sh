#!/bin/bash
# Builds the host JNI library, generates fixtures, runs the Kotlin/JVM bridge tests with Gradle.
#   LITE=1 ./scripts/run_android_jvm_tests.sh    -> library built WITHOUT the tract backend (the Android shipping candidate: no C toolchain needed)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${LITE:-0}" = "1" ]; then
  cargo build --release -p aicore-jni --no-default-features --target-dir target/lite
  export AICORE_JNI_LIB="$PWD/target/lite/release/libaicore_jni.so"
else
  cargo build --release -p aicore-jni
  export AICORE_JNI_LIB="$PWD/target/release/libaicore_jni.so"
fi
python3 scripts/make_android_fixtures.py runs/android_fixtures
export AICORE_FIXTURES="$PWD/runs/android_fixtures"
cd platforms/android/core-bridge
gradle --no-daemon -q test --rerun-tasks "$@"
