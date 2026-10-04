package com.aiagent.core

/**
 * Raw JNI surface of the Rust core (crate `aicore-jni`). Every function returns a JSON document; failures are `{"error": "..."}`.
 * On Android the library is packaged as `libaicore_jni.so` under jniLibs/<abi>/ and loaded with [load]; JVM tests use [loadFrom].
 */
object NativeCore {
    @JvmStatic external fun nativeOpen(root: String, trustJson: String, deviceJson: String): String
    @JvmStatic external fun nativeClose(handle: Long): String
    @JvmStatic external fun nativeImport(handle: Long, path: String): String
    @JvmStatic external fun nativeSolve(handle: Long, intent: String, input: FloatArray): String
    @JvmStatic external fun nativeList(handle: Long): String
    @JvmStatic external fun nativeRunPlan(handle: Long, planJson: String, inputsJson: String): String
    @JvmStatic external fun nativeSetDeviceKey(handle: Long, keyJson: String): String
    @JvmStatic external fun nativeLearn(handle: Long, specJson: String): String
    @JvmStatic external fun nativeAdapt(handle: Long, specJson: String): String
    @JvmStatic external fun nativeAlias(handle: Long, capability: String, keywordsJson: String): String
    @JvmStatic external fun nativePolicyCheck(policyJson: String, requestJson: String, approvalsJson: String): String
    @JvmStatic external fun nativeVersion(): String

    @Volatile private var loaded = false

    @Synchronized fun load(libName: String = "aicore_jni") { if (!loaded) { System.loadLibrary(libName); loaded = true } }
    @Synchronized fun loadFrom(absolutePath: String) { if (!loaded) { System.load(absolutePath); loaded = true } }
}
