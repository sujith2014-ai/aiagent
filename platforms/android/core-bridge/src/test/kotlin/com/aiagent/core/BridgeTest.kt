package com.aiagent.core

import com.sun.net.httpserver.HttpServer
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.net.InetSocketAddress
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.*

/** Runs the real Rust core through the real JNI bridge (host build of the same library that Android packages as libaicore_jni.so). */
class BridgeTest {
    companion object {
        val fx = File(System.getenv("AICORE_FIXTURES") ?: error("AICORE_FIXTURES not set (run scripts/run_android_jvm_tests.sh)"))
        val trust = File(fx, "trust.json").readText()
        val expect = JSONObject(File(fx, "expectations.json").readText())
        val policy = File(fx, "policy.json").readText()
        val caps = listOf("compare_numbers", "point_region", "argmax_position")
        init { NativeCore.loadFrom(System.getenv("AICORE_JNI_LIB") ?: error("AICORE_JNI_LIB not set")) }
        val phone = DeviceProfile.androidLike(listOf("fs.read", "notifications"))
    }

    private fun tmp(): File = kotlin.io.path.createTempDirectory("aicore-test").toFile()
    private fun open(dir: File = tmp(), device: DeviceProfile = phone) = CoreClient.open(File(dir, "rt").absolutePath, trust, device)
    private fun installAll(c: CoreClient) = caps.forEach { assertTrue(c.importCap(File(fx, "caps/$it.cap").absolutePath).activated, it) }

    @Test fun nativeLibraryReportsCoreVersionAndBackend() {
        val v = CoreClient.version()
        assertEquals("0.1.0", v.getString("core")); assertEquals("onnx-mlp-lite", v.getJSONArray("backends").getString(0)); assertEquals("cap/1", v.getJSONArray("package_formats").getString(0))
    }

    @Test fun packagesBuiltOnPcInstallThroughTheFullActivationSequence() {
        open().use { c ->
            val r = c.importCap(File(fx, "caps/compare_numbers.cap").absolutePath)
            assertTrue(r.activated); assertNull(r.failedStep)
            assertEquals(listOf("parse", "signature", "hashes", "compatibility", "sandbox_load", "bundled_tests", "resource_check", "activate").map { "$it:ok" }, r.steps)
            assertTrue(r.testAccuracy!! >= 0.95)
        }
    }

    @Test fun solveMatchesTheRustCliOutputsForTheSamePackages() {
        open().use { c ->
            installAll(c)
            val solves = expect.getJSONArray("solves"); assertEquals(60, solves.length())
            for (i in 0 until solves.length()) {
                val e = solves.getJSONObject(i); val inp = e.getJSONArray("input")
                val r = c.solve(e.getString("intent"), FloatArray(inp.length()) { inp.getDouble(it).toFloat() })
                assertIs<SolveResult.Answer>(r)
                assertEquals(e.getString("capability"), r.capabilityId); assertEquals(e.getString("label"), r.label); assertEquals(e.getInt("label_index"), r.labelIndex)
                assertEquals(e.getString("status"), r.status)
                val pe = e.getJSONArray("probs"); for (k in r.probs.indices) assertTrue(Math.abs(pe.getDouble(k) - r.probs[k]) < 1e-6, "prob $k of case $i")
            }
        }
    }

    @Test fun unknownTasksAndOutOfRangeInputsAreRefusedNotFabricated() {
        open().use { c ->
            installAll(c)
            val u = c.solve("translate this sentence to french", floatArrayOf(0.1f, 0.2f)); assertIs<SolveResult.NeedsHelp>(u)
            val o = c.solve("compare these two numbers", floatArrayOf(5f, 3f)); assertIs<SolveResult.NeedsHelp>(o); assertEquals("OUT_OF_DISTRIBUTION", o.reasonCode)
            assertEquals(expect.getJSONObject("unknown").getString("result"), "NEEDS_HELP"); assertEquals(expect.getJSONObject("ood").getString("reason_code"), "OUT_OF_DISTRIBUTION")
            val wrongShape = c.solve("compare these two numbers", floatArrayOf(0.1f, 0.2f, 0.3f)); assertIs<SolveResult.NeedsHelp>(wrongShape)
        }
    }

    @Test fun hostileAndIncompatiblePackagesAreRejectedAndLeaveNothingBehind() {
        open().use { c ->
            val expected = mapOf("tampered.cap" to "hashes", "untrusted.cap" to "signature", "needs_camera.cap" to "compatibility", "garbage.cap" to "parse")
            for ((f, step) in expected) {
                val r = c.importCap(File(fx, "bad/$f").absolutePath)
                assertFalse(r.activated, f); assertEquals(step, r.failedStep, f)
            }
            assertEquals(0, c.list().length())
            val missing = c.importCap(File(fx, "bad/does-not-exist.cap").absolutePath); assertFalse(missing.activated); assertEquals("parse", missing.failedStep)
        }
    }

    @Test fun deviceSpecificCapabilityIsAcceptedOnlyWhereTheDeviceProvidesIt() {
        open(device = DeviceProfile.androidLike(listOf("fs.read", "notifications", "camera"))).use { c ->
            assertTrue(c.importCap(File(fx, "bad/needs_camera.cap").absolutePath).activated)          // same package, device that has a camera
        }
    }

    @Test fun planExecutionThroughTheBridgeComposesInstalledModules() {
        open().use { c ->
            installAll(c)
            val ins = expect.getJSONObject("plan").getJSONObject("inputs").getJSONArray("v")
            val r = c.runPlan(File(fx, "plan.json").readText(), mapOf("v" to FloatArray(ins.length()) { ins.getDouble(it).toFloat() }))
            val out = r.getJSONArray("outputs"); val want = expect.getJSONObject("plan").getJSONArray("outputs")
            for (i in 0 until want.length()) assertEquals(want.getDouble(i), out.getDouble(i), 1e-6)
            assertEquals(5, r.getJSONObject("capability_calls").getInt("compare_numbers"))
        }
    }

    @Test fun policyChecksRunInTheCoreAndFailClosedOnBadInput() {
        val allow = CoreClient.policyCheck(policy, """{"action":"web.search","params":{"query":"grade band thresholds"}}""")
        assertEquals("allow", allow.getString("effect"))
        for (req in listOf("""{"action":"shell.exec","params":{"cmd":"ls"}}""", """{"action":"web.fetch","params":{"url":"https://169.254.169.254/x"}}""", """{"action":"nope"}"""))
            assertEquals("deny", CoreClient.policyCheck(policy, req).getString("effect"), req)
        assertFailsWith<CoreException> { CoreClient.policyCheck("not json", "{}") }
        assertFailsWith<CoreException> { CoreClient.policyCheck(policy, "{broken") }
    }

    @Test fun guardedDeviceToolsAskThePolicyBeforeTouchingTheDevice() {
        val log = mutableListOf<String>()
        val notify = object : DeviceTool { override val name = "notify"; override val requiredCap = "notifications"; override fun invoke(args: Map<String, String>): String { log += "notify:${args["text"]}"; return "shown" } }
        val camera = object : DeviceTool { override val name = "camera"; override val requiredCap = "camera"; override fun invoke(args: Map<String, String>): String { log += "camera"; return "photo" } }
        val pol = """{"version":1,"default":"deny","rules":[{"id":"allow-notify","action":"device.notify","effect":"allow"},{"id":"camera-needs-approval","action":"device.camera","effect":"require_approval"}]}"""
        val g = GuardedDeviceTools(pol, mapOf("notify" to notify, "camera" to camera))
        assertEquals(listOf("camera", "notifications"), g.capabilities())
        assertEquals(ToolOutcome.Done("shown"), g.invoke("notify", mapOf("text" to "hello")))
        assertIs<ToolOutcome.NeedsApproval>(g.invoke("camera", emptyMap()))
        assertIs<ToolOutcome.Unknown>(g.invoke("microphone", emptyMap()))
        assertIs<ToolOutcome.Denied>(g.invoke("notify", mapOf("text" to "my password=hunter2")))     // secret-looking content is refused before any device call
        assertEquals(listOf("notify:hello"), log)                                                     // the camera and the secret-bearing notification never ran
        val broken = GuardedDeviceTools("garbage", mapOf("notify" to notify)); assertIs<ToolOutcome.Denied>(broken.invoke("notify", mapOf("text" to "x")))   // fail closed
    }

    @Test fun misuseNeverCrashesTheProcess() {
        val c = open(); installAll(c); c.close()
        assertFailsWith<CoreException> { c.solve("compare these two numbers", floatArrayOf(0.1f, 0.2f)) }           // closed client
        assertFailsWith<CoreException> { c.list() }
        c.close()                                                                                                     // double close is harmless
        assertTrue(JSONObject(NativeCore.nativeSolve(987654321L, "x", floatArrayOf(1f))).has("error"))              // unknown handle
        assertTrue(JSONObject(NativeCore.nativeImport(-1L, "x")).has("error"))
        assertFailsWith<CoreException> { CoreClient.open(File(tmp(), "rt").absolutePath, "{broken", phone) }          // bad trust roots
        open().use { c2 ->
            installAll(c2)
            val huge = c2.solve("compare these two numbers", FloatArray(1_000_000) { 0.5f }); assertIs<SolveResult.NeedsHelp>(huge)   // wrong shape, handled
        }
    }

    @Test fun nonFiniteInputsAreRefusedNeverAnsweredWithGarbage() {
        open().use { c ->
            installAll(c)
            for (bad in listOf(floatArrayOf(Float.NaN, 0.5f), floatArrayOf(0.5f, Float.POSITIVE_INFINITY), floatArrayOf(Float.NEGATIVE_INFINITY, Float.NaN))) {
                val r = c.solve("compare these two numbers", bad)
                assertIs<SolveResult.NeedsHelp>(r, "input ${bad.toList()}"); assertEquals("INVALID_INPUT", r.reasonCode)
            }
            val plan = c.runPlan(File(fx, "plan.json").readText(), mapOf("v" to floatArrayOf(0.9f, Float.NaN, 0.7f, 0.4f)))
            assertTrue(plan.optBoolean("needs_help") || plan.has("error"), "a plan must not push NaN through a module: $plan")
        }
    }

    @Test fun manyThreadsShareOneRuntimeSafely() {
        open().use { c ->
            installAll(c)
            val pool = Executors.newFixedThreadPool(8); val ok = java.util.concurrent.atomic.AtomicInteger()
            val solves = expect.getJSONArray("solves")
            repeat(8) { t -> pool.submit { for (i in 0 until 50) {
                val e = solves.getJSONObject((t * 7 + i) % solves.length()); val inp = e.getJSONArray("input")
                val r = c.solve(e.getString("intent"), FloatArray(inp.length()) { inp.getDouble(it).toFloat() })
                if (r is SolveResult.Answer && r.label == e.getString("label")) ok.incrementAndGet()
            } } }
            pool.shutdown(); assertTrue(pool.awaitTermination(60, TimeUnit.SECONDS)); assertEquals(400, ok.get())
        }
    }

    @Test fun registrySurvivesCloseAndReopenAndAFreshInstallReproducesTheSameOutputs() {
        val dir = tmp()
        open(dir).use { installAll(it) }
        open(dir).use { c ->                                                             // reopen: capabilities persist, no re-import needed
            assertEquals(3, c.list().length())
            val e = expect.getJSONArray("solves").getJSONObject(0); val inp = e.getJSONArray("input")
            val r = c.solve(e.getString("intent"), FloatArray(inp.length()) { inp.getDouble(it).toFloat() }) as SolveResult.Answer
            assertEquals(e.getString("label"), r.label)
        }
        val fresh = tmp()                                                                // fresh installation from the same files
        open(fresh).use { c -> installAll(c); val e = expect.getJSONArray("solves").getJSONObject(30); val inp = e.getJSONArray("input")
            assertEquals(e.getString("label"), (c.solve(e.getString("intent"), FloatArray(inp.length()) { inp.getDouble(it).toFloat() }) as SolveResult.Answer).label) }
    }

    private fun serve(body: ByteArray?, status: Int = 200, declared: Long? = null): Pair<HttpServer, String> {
        val s = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
        s.createContext("/") { ex -> val b = body ?: ByteArray(0); ex.sendResponseHeaders(status, if (declared != null) declared else b.size.toLong()); ex.responseBody.use { it.write(b) } }
        s.start(); return s to "http://127.0.0.1:${s.address.port}/x.cap"
    }

    @Test fun serverToDeviceDeliveryInstallsGoodPackagesAndRejectsBadOnesAndCleansUp() {
        val dir = tmp(); val temp = File(dir, "dl")
        open(dir).use { c ->
            val inst = CapInstaller(c, temp, maxBytes = 1 shl 20)
            val good = File(fx, "caps/point_region.cap").readBytes()
            val (s1, u1) = serve(good); val sha = java.security.MessageDigest.getInstance("SHA-256").digest(good).joinToString("") { "%02x".format(it) }
            assertIs<CapInstaller.Result.Installed>(inst.installFromUrl(u1, sha)); s1.stop(0)
            assertEquals(1, c.list().length())
            val (s2, u2) = serve(good); assertIs<CapInstaller.Result.DownloadFailed>(inst.installFromUrl(u2, "0".repeat(64))); s2.stop(0)          // pinned hash mismatch
            val (s3, u3) = serve(File(fx, "bad/tampered.cap").readBytes()); assertIs<CapInstaller.Result.Rejected>(inst.installFromUrl(u3)).also { assertEquals("hashes", it.report.failedStep) }; s3.stop(0)
            val (s4, u4) = serve(File(fx, "bad/untrusted.cap").readBytes()); assertIs<CapInstaller.Result.Rejected>(inst.installFromUrl(u4)).also { assertEquals("signature", it.report.failedStep) }; s4.stop(0)
            val (s5, u5) = serve(ByteArray(0), 404); assertIs<CapInstaller.Result.DownloadFailed>(inst.installFromUrl(u5)); s5.stop(0)
            val (s6, u6) = serve(ByteArray(3 shl 20) { 1 }); assertIs<CapInstaller.Result.DownloadFailed>(inst.installFromUrl(u6)); s6.stop(0)                    // over the size cap
            assertIs<CapInstaller.Result.DownloadFailed>(inst.installFromUrl("http://127.0.0.1:1/never.cap"))                                                          // unreachable
            assertEquals(1, c.list().length())                                                                                                                         // only the good package
            assertEquals(0, temp.listFiles()!!.size)                                                                                                                   // no leftovers
        }
    }

    private fun learnSpec(id: String = "larger_of_two", minAcc: Double = 0.9): String {
        val rnd = java.util.Random(7); val x = JSONArray(); val y = JSONArray()
        repeat(240) { val a = rnd.nextDouble() * 10; val b = rnd.nextDouble() * 10; x.put(JSONArray(listOf(a, b))); y.put(if (a > b) 1 else 0) }
        return JSONObject().put("capability_id", id).put("description", "which of two numbers is larger").put("keywords", JSONArray(listOf("larger", "bigger", "numbers")))
            .put("labels", JSONArray(listOf("second", "first"))).put("x", x).put("y", y).put("min_accuracy", minAcc).toString()
    }

    @Test fun onDeviceLearningThroughTheBridgeSignsWithTheDeviceKeyAndIsUsableOnlyWhileThatKeyIsSet() {
        val dir = tmp(); val seed = "11".repeat(32)
        open(dir).use { c ->
            // no device key: the core refuses cleanly, nothing installed
            assertTrue(runCatching { c.learn(learnSpec()) }.isFailure)
            assertTrue(c.setDeviceKey("phone-1", seed).isNotEmpty())
            val r = c.learn(learnSpec())
            assertTrue(r.getBoolean("learned"), r.toString()); assertTrue(r.getDouble("test_accuracy") >= 0.9)
            val ok = c.solve("which of two numbers is larger", floatArrayOf(7f, 2f)); assertIs<SolveResult.Answer>(ok); assertEquals("first", ok.label)
            val ok2 = c.solve("which of two numbers is larger", floatArrayOf(1f, 9f)); assertIs<SolveResult.Answer>(ok2); assertEquals("second", ok2.label)
            // router update on the device bumps the version and keeps the model
            assertTrue(c.alias("larger_of_two", listOf("greater")).getBoolean("activated"))
            assertIs<SolveResult.Answer>(c.solve("which is greater", floatArrayOf(7f, 2f)))
        }
        // restart WITHOUT the device key: the stored package is signed by a key this runtime no longer trusts -> refused, not run
        open(dir).use { c ->
            val r = c.solve("which of two numbers is larger", floatArrayOf(7f, 2f)); assertIs<SolveResult.NeedsHelp>(r); assertEquals("MODEL_UNAVAILABLE", r.reasonCode)
            c.setDeviceKey("phone-1", seed)
            assertIs<SolveResult.Answer>(c.solve("which of two numbers is larger", floatArrayOf(7f, 2f)))
        }
    }

    @Test fun onDeviceLearningRefusesBadRequestsAndRandomLabelsThroughTheBridge() {
        open().use { c ->
            c.setDeviceKey("phone-1", "22".repeat(32))
            assertTrue(runCatching { c.learn(JSONObject(learnSpec()).put("capability_id", "bad id!").toString()) }.isFailure)
            val noise = JSONObject(learnSpec()); val y = noise.getJSONArray("y"); val rnd = java.util.Random(1); for (k in 0 until y.length()) y.put(k, rnd.nextInt(2))
            val r = c.learn(noise.toString()); assertFalse(r.getBoolean("learned")); assertTrue(r.getString("reason").contains("held-out accuracy"))
            assertEquals(0, c.list().length())
        }
    }
}
