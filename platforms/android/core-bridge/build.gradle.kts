// Pure Kotlin/JVM module: no Android SDK classes. It runs unchanged on Android (org.json comes from the platform there) and on the JVM,
// where the tests load the host build of the same Rust JNI library. Android-specific code lives in ../app (not compiled in CI here).
plugins { kotlin("jvm") version "2.0.21" }

repositories { mavenCentral() }

dependencies {
    compileOnly("org.json:json:20240303")
    testImplementation("org.json:json:20240303")
    testImplementation(kotlin("test"))
}

kotlin { jvmToolchain(21) }

tasks.test {
    useJUnitPlatform()
    // set by scripts/run_android_jvm_tests.sh
    environment("AICORE_JNI_LIB", System.getenv("AICORE_JNI_LIB") ?: "")
    environment("AICORE_FIXTURES", System.getenv("AICORE_FIXTURES") ?: "")
    testLogging { events("passed", "failed", "skipped"); showStandardStreams = false }
}
