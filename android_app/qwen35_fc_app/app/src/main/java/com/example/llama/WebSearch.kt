package com.example.llama

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

/**
 * Minimal on-device "web_search" tool implementation.
 *
 * Tries DuckDuckGo Instant Answers first, then falls back to Wikipedia
 * search snippets. No API key required.
 */
object WebSearch {
    private val TAG = WebSearch::class.java.simpleName
    private const val TIMEOUT_MS = 15000

    suspend fun search(query: String): String = withContext(Dispatchers.IO) {
        val clean = query.trim().take(300)
        if (clean.isEmpty()) return@withContext "No query provided."
        try {
            val ddg = duckDuckGo(clean)
            if (ddg != null) return@withContext ddg
            val wiki = wikipedia(clean)
            if (wiki != null) return@withContext wiki
            "No web results found for \"$clean\"."
        } catch (e: Exception) {
            Log.e(TAG, "web_search failed", e)
            "Web search failed: ${e.message}"
        }
    }

    private fun duckDuckGo(query: String): String? {
        val url =
            "https://api.duckduckgo.com/?q=${URLEncoder.encode(query, "UTF-8")}" +
                "&format=json&no_html=1&skip_disambig=1"
        val json = JSONObject(get(url))
        val out = StringBuilder()
        val abstract = json.optString("AbstractText").trim()
        val source = json.optString("AbstractSource").trim()
        val heading = json.optString("Heading").trim()
        if (abstract.isNotEmpty()) {
            out.append("Top answer")
            if (heading.isNotEmpty()) out.append(" ($heading)")
            if (source.isNotEmpty()) out.append(" [source: $source]")
            out.append(": ").append(abstract).append("\n")
        }
        val related = json.optJSONArray("RelatedTopics")
        var added = 0
        if (related != null) {
            var i = 0
            while (i < related.length() && added < 3) {
                val item = related.optJSONObject(i)
                val text = item?.optString("Text")?.trim().orEmpty()
                if (text.isNotEmpty()) {
                    added++
                    out.append("$added. ").append(text).append("\n")
                }
                i++
            }
        }
        val result = out.toString().trim()
        return if (result.length > 40) "Web results for \"$query\":\n$result" else null
    }

    private fun wikipedia(query: String): String? {
        val url =
            "https://en.wikipedia.org/w/api.php?action=query&list=search" +
                "&srsearch=${URLEncoder.encode(query, "UTF-8")}" +
                "&srlimit=3&format=json"
        val json = JSONObject(get(url))
        val results = json.optJSONObject("query")?.optJSONArray("search") ?: return null
        if (results.length() == 0) return null
        val out = StringBuilder("Wikipedia results for \"$query\":\n")
        for (i in 0 until results.length()) {
            val item = results.getJSONObject(i)
            val title = item.optString("title")
            val snippet = item.optString("snippet")
                .replace(Regex("<[^>]*>"), "")
                .replace("&quot;", "\"")
                .replace("&#039;", "'")
                .replace("&amp;", "&")
                .trim()
            out.append("${i + 1}. $title: $snippet\n")
        }
        return out.toString().trim()
    }

    private fun get(urlString: String): String {
        val conn = (URL(urlString).openConnection() as HttpURLConnection).apply {
            connectTimeout = TIMEOUT_MS
            readTimeout = TIMEOUT_MS
            setRequestProperty("User-Agent", "QwenAndroidApp/1.0")
            setRequestProperty("Accept", "application/json")
        }
        try {
            val code = conn.responseCode
            if (code !in 200..299) throw IllegalStateException("HTTP $code")
            return conn.inputStream.bufferedReader().use { it.readText() }.take(200_000)
        } finally {
            conn.disconnect()
        }
    }
}
