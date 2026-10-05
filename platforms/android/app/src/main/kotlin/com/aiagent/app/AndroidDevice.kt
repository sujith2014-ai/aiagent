package com.aiagent.app

import android.app.ActivityManager
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import com.aiagent.core.DeviceProfile
import com.aiagent.core.DeviceTool

/** Builds the core's DeviceProfile from what this phone actually has. UNVALIDATED on a device. */
object AndroidDevice {
    fun profile(ctx: Context): DeviceProfile {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val totalMb = mi.totalMem / (1024 * 1024)
        val caps = mutableListOf("clipboard")
        if (Build.VERSION.SDK_INT < 33 || ctx.checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) == PackageManager.PERMISSION_GRANTED) caps += "notifications"
        // camera/microphone/filesystem are deliberately absent: the manifest does not request them.
        return DeviceProfile("ANDROID_${Build.MODEL}", caps, maxRamMb = minOf(512L, totalMb / 8), maxModelBytes = 8L shl 20, maxActiveModules = if (am.isLowRamDevice) 2 else 4)
    }
}

class NotifyTool(private val ctx: Context) : DeviceTool {
    override val name = "notify"; override val requiredCap = "notifications"
    override fun invoke(args: Map<String, String>): String {
        val nm = ctx.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        nm.createNotificationChannel(NotificationChannel("aiagent", "aiagent", NotificationManager.IMPORTANCE_DEFAULT))
        nm.notify(1, NotificationCompat.Builder(ctx, "aiagent").setSmallIcon(android.R.drawable.ic_dialog_info).setContentTitle("aiagent").setContentText(args["text"] ?: "").build())
        return "shown"
    }
}

class ClipboardTool(private val ctx: Context) : DeviceTool {
    override val name = "clipboard.copy"; override val requiredCap = "clipboard"
    override fun invoke(args: Map<String, String>): String {
        (ctx.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager).setPrimaryClip(ClipData.newPlainText("aiagent", args["text"] ?: ""))
        return "copied"
    }
}
