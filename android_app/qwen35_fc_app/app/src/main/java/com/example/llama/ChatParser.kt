package com.example.llama

/**
 * Parses raw model output into displayable parts:
 * - <think>...</think> reasoning shown in a "Thought" card
 * - <tool_call>...</tool_call> blocks (Qwen XML/JSON style) executed as tools,
 *   never shown raw to the user
 * - remaining text shown as the final answer
 */
object ChatParser {

    data class ToolCall(val name: String, val query: String, val raw: String)

    data class Parsed(
        val thinking: String?,
        val toolCall: ToolCall?,
        val answer: String,
    )

    private val thinkRegex = Regex("<think>(.*?)</think>", RegexOption.DOT_MATCHES_ALL)
    private val unclosedThinkRegex = Regex("<think>(.*)$", RegexOption.DOT_MATCHES_ALL)
    private val headerEchoRegex = Regex("^\\s*<\\|im_start\\|>\\s*assistant\\s*")
    private val strayTagRegex = Regex("<\\|im_start\\|>|<\\|im_end\\|>")
    private val strayRoleRegex = Regex("^(assistant|user|system)\\s*")
    private val toolRegex = Regex("<tool_call>(.*?)</tool_call>", RegexOption.DOT_MATCHES_ALL)
    private val functionTagRegex = Regex("<function=([^>]+)>")
    private val nameJsonRegex = Regex("\"name\"\\s*:\\s*\"([^\"]+)\"")
    private val queryJsonRegex = Regex("\"query\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"")
    private val queryParamRegex =
        Regex("<parameter=query>(.*?)</parameter>", RegexOption.DOT_MATCHES_ALL)
    // Bare native calls the fine-tuned model emits without <tool_call> tags,
    // e.g. web_search({"query": "..."}) — possibly unbalanced/malformed.
    private val bareCallRegex =
        Regex("(web_search|google_search|search)\\s*\\((.*)\\)\\s*$", RegexOption.DOT_MATCHES_ALL)
    private val bareCallInlineRegex =
        Regex("(web_search|google_search|search)\\s*\\(\\s*(\\{.*\\})\\s*\\)", RegexOption.DOT_MATCHES_ALL)
    private val toolResponseRegex =
        Regex("<tool_response>.*?(</tool_response>|$)", RegexOption.DOT_MATCHES_ALL)

    fun parse(raw: String): Parsed {
        var text = raw
        val hadHeaderEcho = headerEchoRegex.containsMatchIn(text)
        text = headerEchoRegex.replace(text, "")
        if (hadHeaderEcho) text = strayRoleRegex.replace(text, "")

        // Tools first: a call may sit inside (or after) a think block.
        var toolCall: ToolCall? = null
        toolRegex.find(text)?.let { match ->
            toolCall = parseToolInner(match.groupValues[1], match.value)
            text = text.replace(match.value, "")
        }
        if (toolCall == null) {
            // Try bare `web_search({...})` style, anchored at the end first.
            bareCallRegex.find(text.trim())?.let { match ->
                toolCall = parseToolInner(match.groupValues[2], match.value)
                text = text.replace(match.value, "")
            } ?: bareCallInlineRegex.find(text)?.let { match ->
                toolCall = parseToolInner(match.groupValues[2], match.value)
                text = text.replace(match.value, "")
            }
        }
        if (toolCall == null && text.contains("<tool_call>")) {
            // Tolerate a MISSING </tool_call>: the model sometimes emits an
            // open tag, JSON, then continues into <tool_response> or stops.
            val start = text.indexOf("<tool_call>")
            val respIdx = text.indexOf("<tool_response>", start)
            val end = if (respIdx >= 0) respIdx else text.length
            val inner = text.substring(start + "<tool_call>".length, end)
            if (nameJsonRegex.containsMatchIn(inner) ||
                functionTagRegex.containsMatchIn(inner) ||
                queryJsonRegex.containsMatchIn(inner)
            ) {
                toolCall = parseToolInner(inner, text.substring(start, end))
            }
            text = text.substring(0, start) +
                (if (respIdx >= 0) text.substring(respIdx) else "")
        }
        // The model sometimes hallucinates its own (empty) <tool_response>
        // section while emitting the call — always drop assistant-side ones.
        text = toolResponseRegex.replace(text, "")

        // Thinking second (closed blocks, then unclosed trailing ones).
        var thinking = thinkRegex.find(text)?.groupValues?.get(1)?.trim().orEmpty()
            .takeIf { it.isNotEmpty() }
            ?: unclosedThinkRegex.find(text)?.groupValues?.get(1)?.trim()
                .orEmpty().takeIf { it.isNotEmpty() }
        text = thinkRegex.replace(text, "")
        text = unclosedThinkRegex.replace(text, "")
        if (thinking == null && text.contains("</think>")) {
            // Lone closer: the <think> opener lives in the prompt prefix, so
            // everything before the first </think> is the reasoning trace.
            val end = text.indexOf("</think>")
            thinking = text.substring(0, end).trim().takeIf { it.isNotEmpty() }
            text = text.substring(end + "</think>".length)
        }

        // Drop any other leftover markup tags for a clean chat answer.
        text = text
            .replace(strayTagRegex, "")
            .replace(Regex("</?think>"), "")
            .replace(Regex("</?tool_call>"), "")
            .replace(Regex("</?tool_response>"), "")
            .replace(Regex("</?function[^>]*>"), "")
            .replace(Regex("</?parameter[^>]*>"), "")
            // Last-resort sweep: never show a raw function-call invocation.
            .replace(bareCallInlineRegex, "")
            .trim()
        return Parsed(thinking, toolCall, text)
    }

    private fun parseToolInner(inner: String, raw: String): ToolCall {
        val name = functionTagRegex.find(inner)?.groupValues?.get(1)?.trim()
            ?: nameJsonRegex.find(inner)?.groupValues?.get(1)?.trim()
            ?: "web_search"
        val query = queryParamRegex.find(inner)?.groupValues?.get(1)?.trim()
            ?: queryJsonRegex.find(inner)?.groupValues?.get(1)
                ?.replace("\\\"", "\"")?.replace("\\n", " ")?.trim()
            // Fallback: treat the whole inner text as the query.
            ?: inner.trim().trim('{', '}', '(', ')', '"', '\'').trim().take(300)
        return ToolCall(canonicalToolName(name), query, raw)
    }

    private fun canonicalToolName(name: String): String {
        val n = name.lowercase()
        return when {
            "search" in n -> "web_search"
            else -> name
        }
    }
}
