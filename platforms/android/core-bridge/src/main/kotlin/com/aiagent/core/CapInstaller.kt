package com.aiagent.core

import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Server -> device capability delivery: download to a private temp file (size-capped, optional SHA-256 pin), hand it to the core's full activation
 * sequence (signature, hashes, compatibility, sandbox load, bundled tests, resource check), and always delete the temp file.
 * Nothing is executed or registered unless the core activates it.
 */
class CapInstaller(private val client: CoreClient, private val tempDir: File, private val maxBytes: Long = 64L shl 20, private val timeoutMs: Int = 15000) {
    sealed class Result {
        data class Installed(val report: ImportReport) : Result()
        data class Rejected(val report: ImportReport) : Result()
        data class DownloadFailed(val reason: String) : Result()
    }

    fun installFromUrl(url: String, expectedSha256: String? = null): Result {
        tempDir.mkdirs()
        val tmp = File.createTempFile("cap-", ".cap", tempDir)
        try {
            val conn = URL(url).openConnection() as HttpURLConnection
            conn.connectTimeout = timeoutMs; conn.readTimeout = timeoutMs; conn.instanceFollowRedirects = false
            if (conn.responseCode != 200) return Result.DownloadFailed("HTTP ${conn.responseCode}")
            val declared = conn.contentLengthLong
            if (declared > maxBytes) return Result.DownloadFailed("declared size $declared exceeds limit $maxBytes")
            val md = MessageDigest.getInstance("SHA-256"); var total = 0L
            conn.inputStream.use { ins -> tmp.outputStream().use { outs ->
                val buf = ByteArray(16 * 1024)
                while (true) {
                    val n = ins.read(buf); if (n < 0) break
                    total += n; if (total > maxBytes) return Result.DownloadFailed("download exceeds limit $maxBytes")
                    md.update(buf, 0, n); outs.write(buf, 0, n)
                }
            } }
            val sha = md.digest().joinToString("") { "%02x".format(it) }
            if (expectedSha256 != null && !sha.equals(expectedSha256, ignoreCase = true)) return Result.DownloadFailed("sha256 mismatch (pinned $expectedSha256, got $sha)")
            val rep = client.importCap(tmp.absolutePath)
            return if (rep.activated) Result.Installed(rep) else Result.Rejected(rep)
        } catch (e: java.io.IOException) {
            return Result.DownloadFailed(e.message ?: e.javaClass.simpleName)
        } finally { tmp.delete() }
    }
}
