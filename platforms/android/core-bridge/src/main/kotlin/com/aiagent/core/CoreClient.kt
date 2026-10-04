package com.aiagent.core

import org.json.JSONArray
import org.json.JSONObject

/** What this device offers. The only place platform differences enter the core (mirrors the Rust `DeviceProfile`). */
data class DeviceProfile(val name: String, val caps: List<String>, val maxRamMb: Long, val maxModelBytes: Long, val maxActiveModules: Int) {
    fun toJson(): String = JSONObject().put("name", name).put("caps", JSONArray(caps)).put("max_ram_mb", maxRamMb)
        .put("max_model_bytes", maxModelBytes).put("max_active_modules", maxActiveModules).toString()

    companion object {
        /** Android-like limits; the real values come from ActivityManager/PackageManager in the app module. */
        fun androidLike(caps: List<String>, ramMb: Long = 512, maxModelBytes: Long = 8L shl 20, maxActiveModules: Int = 4) =
            DeviceProfile("ANDROID_LIKE", caps, ramMb, maxModelBytes, maxActiveModules)
    }
}

class CoreException(message: String) : RuntimeException(message)

data class ImportReport(val activated: Boolean, val capabilityId: String?, val version: String?, val failedStep: String?, val steps: List<String>, val testAccuracy: Double?, val raw: JSONObject)

sealed class SolveResult {
    data class Answer(val capabilityId: String, val version: String, val label: String, val labelIndex: Int, val probs: List<Float>, val confidence: Float, val status: String, val flags: List<String>, val raw: JSONObject) : SolveResult()
    data class NeedsHelp(val reasonCode: String, val reason: String, val raw: JSONObject) : SolveResult()
}

private fun JSONObject.orThrow(): JSONObject { if (has("error")) throw CoreException(getString("error")); return this }
private fun JSONArray.strings(): List<String> = (0 until length()).map { getString(it) }

/** Typed, thread-safe wrapper around one native runtime handle. Never throws for domain outcomes (refusals, rejected packages); throws [CoreException] only for misuse. */
class CoreClient private constructor(private var handle: Long) : AutoCloseable {
    private val lock = Any()

    companion object {
        fun open(root: String, trustJson: String, device: DeviceProfile): CoreClient {
            val r = JSONObject(NativeCore.nativeOpen(root, trustJson, device.toJson())).orThrow()
            return CoreClient(r.getLong("handle"))
        }
        fun version(): JSONObject = JSONObject(NativeCore.nativeVersion())
        fun policyCheck(policyJson: String, requestJson: String, approvalsJson: String = "[]"): JSONObject =
            JSONObject(NativeCore.nativePolicyCheck(policyJson, requestJson, approvalsJson)).orThrow()
    }

    private fun h(): Long = synchronized(lock) { if (handle == 0L) throw CoreException("client is closed") else handle }

    fun importCap(path: String): ImportReport {
        val j = JSONObject(NativeCore.nativeImport(h(), path)).orThrow()
        val steps = j.getJSONArray("steps"); val names = (0 until steps.length()).map { steps.getJSONObject(it).getString("step") + if (steps.getJSONObject(it).getBoolean("ok")) ":ok" else ":FAILED" }
        return ImportReport(j.getBoolean("activated"), j.optString("capability_id").ifEmpty { null }, j.optString("version").ifEmpty { null },
            if (j.isNull("failed_step")) null else j.getString("failed_step"), names, if (j.isNull("test_accuracy")) null else j.getDouble("test_accuracy"), j)
    }

    /** Install the device-local signing key (public part returned). Learned/adapted packages are signed with it and trusted by this runtime only while it is set. */
    fun setDeviceKey(keyId: String, seedHex: String): String =
        JSONObject(NativeCore.nativeSetDeviceKey(h(), JSONObject().put("key_id", keyId).put("seed_hex", seedHex).toString())).orThrow().getString("public_b64")

    /** Train a new capability on the device from labelled examples. [specJson] has the CLI `learn` spec shape. Refusals come back in the report (learned=false), not as exceptions. */
    fun learn(specJson: String): JSONObject = JSONObject(NativeCore.nativeLearn(h(), specJson)).orThrow()
    fun adapt(specJson: String): JSONObject = JSONObject(NativeCore.nativeAdapt(h(), specJson)).orThrow()
    fun alias(capability: String, keywords: List<String>): JSONObject =
        JSONObject(NativeCore.nativeAlias(h(), capability, org.json.JSONArray(keywords).toString())).orThrow()

    fun solve(intent: String, input: FloatArray): SolveResult {
        val j = JSONObject(NativeCore.nativeSolve(h(), intent, input)).orThrow()
        return if (j.getString("result") == "ANSWER") {
            val p = j.getJSONArray("probs"); val f = j.getJSONArray("flags")
            SolveResult.Answer(j.getString("capability_id"), j.getString("version"), j.getString("label"), j.getInt("label_index"), (0 until p.length()).map { p.getDouble(it).toFloat() },
                j.getDouble("confidence").toFloat(), j.getString("status"), f.strings(), j)
        } else SolveResult.NeedsHelp(j.getString("reason_code"), j.getString("reason"), j)
    }

    fun list(): JSONArray = org.json.JSONArray(NativeCore.nativeList(h()).also { if (it.startsWith("{")) JSONObject(it).orThrow() })

    fun runPlan(planJson: String, inputs: Map<String, FloatArray>): JSONObject {
        // JSON cannot carry NaN/Infinity; refuse here exactly as the core does for solve() instead of failing inside the serializer
        if (inputs.values.any { a -> a.any { !it.isFinite() } }) return JSONObject().put("needs_help", true).put("reason", "INVALID_INPUT: non-finite value in plan inputs")
        val ins = JSONObject(); inputs.forEach { (k, v) -> ins.put(k, JSONArray(v.toList())) }
        return JSONObject(NativeCore.nativeRunPlan(h(), planJson, ins.toString())).orThrow()
    }

    override fun close() { synchronized(lock) { if (handle != 0L) { NativeCore.nativeClose(handle); handle = 0L } } }
}
