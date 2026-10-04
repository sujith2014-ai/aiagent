package com.aiagent.core

import org.json.JSONObject

/** One device capability implementation (camera, notifications, clipboard, ...). Android implementations live in the app module. */
interface DeviceTool {
    val name: String            // e.g. "notify"
    val requiredCap: String     // abstract capability the core knows about, e.g. "notifications"
    fun invoke(args: Map<String, String>): String
}

sealed class ToolOutcome {
    data class Done(val output: String) : ToolOutcome()
    data class Denied(val ruleId: String, val reason: String) : ToolOutcome()
    data class NeedsApproval(val reason: String) : ToolOutcome()
    data class Unknown(val name: String) : ToolOutcome()
    data class Failed(val message: String) : ToolOutcome()
}

/**
 * Every device action goes through the core's deterministic policy first (action id `device.<tool>`); the AI side only ever proposes.
 * Approvals are supplied by the host app from its own UI, never by model output.
 */
class GuardedDeviceTools(private val policyJson: String, private val tools: Map<String, DeviceTool>, private val approvals: () -> String = { "[]" }) {
    fun capabilities(): List<String> = tools.values.map { it.requiredCap }.distinct().sorted()

    fun invoke(name: String, args: Map<String, String>): ToolOutcome {
        val tool = tools[name] ?: return ToolOutcome.Unknown(name)
        val req = JSONObject().put("action", "device.$name").put("params", JSONObject(args as Map<*, *>)).toString()
        val d = try { CoreClient.policyCheck(policyJson, req, approvals()) } catch (e: CoreException) { return ToolOutcome.Denied("policy-engine-error", e.message ?: "") }   // fail closed
        return when (d.getString("effect")) {
            "allow" -> try { ToolOutcome.Done(tool.invoke(args)) } catch (e: Exception) { ToolOutcome.Failed(e.message ?: e.javaClass.simpleName) }
            "require_approval" -> ToolOutcome.NeedsApproval(d.getString("reason"))
            else -> ToolOutcome.Denied(d.getString("rule_id"), d.getString("reason"))
        }
    }
}
