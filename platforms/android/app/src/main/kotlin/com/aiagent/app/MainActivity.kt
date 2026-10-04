package com.aiagent.app

import android.app.Activity
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import com.aiagent.core.*
import java.io.File
import kotlin.concurrent.thread

/**
 * Minimal shell (UI, lifecycle, storage paths, device profile). All logic - registry, package validation, routing, execution, policy - is in the Rust core.
 * UNVALIDATED: not compiled or run on a device in this repository (no Android SDK reachable). See docs/ANDROID.md and docs/RUNBOOKS.md R2.
 */
class MainActivity : Activity() {
    private lateinit var core: CoreClient
    private lateinit var out: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        NativeCore.load()
        val trust = assets.open("trust.json").bufferedReader().use { it.readText() }          // public keys only; the signing key never lives on the device
        core = CoreClient.open(File(filesDir, "runtime").absolutePath, trust, AndroidDevice.profile(this))
        val url = EditText(this).apply { hint = "https://server/cap/compare_numbers.cap"; setSingleLine() }
        val intent = EditText(this).apply { hint = "intent, e.g. compare these two numbers"; setSingleLine() }
        val input = EditText(this).apply { hint = "numbers, e.g. 0.2, 0.9"; setSingleLine() }
        out = TextView(this)
        val install = Button(this).apply { text = "Install capability"; setOnClickListener { thread { render { installFrom(url.text.toString()) } } } }
        val solve = Button(this).apply { text = "Solve"; setOnClickListener { thread { render { solve(intent.text.toString(), input.text.toString()) } } } }
        setContentView(LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(32, 32, 32, 32); listOf(url, install, intent, input, solve, out).forEach { addView(it) } })
        render { "capabilities: " + core.list().toString() }
    }

    private fun installFrom(url: String): String = when (val r = CapInstaller(core, File(cacheDir, "downloads")).installFromUrl(url)) {
        is CapInstaller.Result.Installed -> "installed ${r.report.capabilityId} ${r.report.version}"
        is CapInstaller.Result.Rejected -> "rejected at ${r.report.failedStep}: ${r.report.steps}"
        is CapInstaller.Result.DownloadFailed -> "download failed: ${r.reason}"
    }

    private fun solve(intent: String, input: String): String {
        val x = try { input.split(",").map { it.trim().toFloat() }.toFloatArray() } catch (e: NumberFormatException) { return "input must be comma-separated numbers" }
        return when (val r = core.solve(intent, x)) {
            is SolveResult.Answer -> "${r.label} (${r.status}, confidence ${"%.2f".format(r.confidence)}) via ${r.capabilityId} ${r.version}"
            is SolveResult.NeedsHelp -> "NEEDS_HELP: ${r.reasonCode} - ${r.reason}"      // never a guess; escalation to a server/teacher is a separate, policy-gated step
        }
    }

    private fun render(f: () -> String) { val s = try { f() } catch (e: Exception) { "error: ${e.message}" }; runOnUiThread { out.text = s } }
    override fun onDestroy() { core.close(); super.onDestroy() }
}
