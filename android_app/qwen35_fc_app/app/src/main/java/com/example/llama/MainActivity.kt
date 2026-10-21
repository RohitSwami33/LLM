package com.example.llama

import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.activity.addCallback
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.arm.aichat.gguf.GgufMetadata
import com.arm.aichat.gguf.GgufMetadataReader
import com.google.android.material.floatingactionbutton.FloatingActionButton
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.UUID

class MainActivity : AppCompatActivity() {

    // Android views
    private lateinit var statusTv: TextView
    private lateinit var messagesRv: RecyclerView
    private lateinit var userInputEt: EditText
    private lateinit var userActionFab: FloatingActionButton
    private lateinit var newChatBtn: Button
    private lateinit var themeToggleBtn: Button

    // Arm AI Chat inference engine
    private lateinit var engine: InferenceEngine
    private var generationJob: Job? = null

    // Conversation states
    private var isModelReady = false
    private val messages: MutableList<Message> = retainedMessages
    private val messageAdapter = MessageAdapter(messages, onThinkingToggle = { id ->
        val index = messages.indexOfFirst { it.id == id }
        if (index >= 0) {
            val msg = messages[index]
            messages[index] = msg.copy(thinkingExpanded = !msg.thinkingExpanded)
            messagesRv.adapter?.notifyItemChanged(index)
        }
    })

    override fun onCreate(savedInstanceState: Bundle?) {
        applySavedTheme()
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        // View model boilerplate and state management is out of this basic sample's scope
        onBackPressedDispatcher.addCallback { Log.w(TAG, "Ignore back press for simplicity") }

        // Find views
        statusTv = findViewById(R.id.model_status)
        messagesRv = findViewById(R.id.messages)
        messagesRv.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        messagesRv.adapter = messageAdapter
        if (messages.isEmpty()) restoreMessages()
        if (messages.isNotEmpty()) messageAdapter.notifyDataSetChanged()
        userInputEt = findViewById(R.id.user_input)
        userActionFab = findViewById(R.id.fab)
        newChatBtn = findViewById(R.id.new_chat)
        themeToggleBtn = findViewById(R.id.theme_toggle)
        updateThemeToggleIcon()

        themeToggleBtn.setOnClickListener {
            val prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            val dark = !isDarkMode()
            prefs.edit().putBoolean(KEY_DARK_MODE, dark).apply()
            AppCompatDelegate.setDefaultNightMode(
                if (dark) AppCompatDelegate.MODE_NIGHT_YES
                else AppCompatDelegate.MODE_NIGHT_NO
            )
            recreate()
        }

        newChatBtn.setOnClickListener { confirmNewChat() }

        // Arm AI Chat initialization
        lifecycleScope.launch(Dispatchers.Default) {
            engine = AiChat.getInferenceEngine(applicationContext)
            if (engine.state.value is InferenceEngine.State.ModelReady &&
                File(ensureModelsDirectory(), POST_TRAINED_MODEL_FILENAME).exists()
            ) {
                // Activity recreated (e.g. theme change): model still loaded.
                withContext(Dispatchers.Main) { markModelReady("Qwen3.5 ORPO-v4") }
            } else {
                try {
                    ensureBundledModelExtracted()
                } catch (e: Exception) {
                    // Extraction failure keeps the app usable: the manual
                    // GGUF picker path still exists.
                    Log.e(TAG, "Bundled model extraction failed", e)
                    withContext(Dispatchers.Main) {
                        setStatus("● Bundled model failed, use picker or reinstall")
                    }
                }
                autoLoadPostTrainedModel()
            }
        }

        // Upon CTA button tapped
        userActionFab.setOnClickListener {
            if (isModelReady) {
                // If model is ready, validate input and send to engine
                handleUserInput()
            } else {
                // Otherwise, prompt user to select a GGUF metadata on the device
                getContent.launch(arrayOf("*/*"))
            }
        }

        // Keyboard send action sends the message like the FAB does.
        userInputEt.setOnEditorActionListener { _, actionId, _ ->
            if (actionId == EditorInfo.IME_ACTION_SEND) {
                if (isModelReady) handleUserInput() else {
                    Toast.makeText(this, "Model is still loading", Toast.LENGTH_SHORT).show()
                }
                true
            } else {
                false
            }
        }
    }

    private val getContent = registerForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri: Uri? ->
        Log.i(TAG, "Selected file uri:\n $uri")
        uri?.let { handleSelectedModel(it) }
    }

    /**
     * Handles the file Uri from [getContent] result
     */
    private fun handleSelectedModel(uri: Uri) {
        // Update UI states
        userActionFab.isEnabled = false
        userInputEt.hint = "Parsing GGUF..."
        setStatus("Parsing metadata from selected file…")

        lifecycleScope.launch(Dispatchers.IO) {
            // Parse GGUF metadata
            Log.i(TAG, "Parsing GGUF metadata...")
            contentResolver.openInputStream(uri)?.use {
                GgufMetadataReader.create().readStructuredMetadata(it)
            }?.let { metadata ->
                // Update UI to show GGUF metadata to user
                Log.i(TAG, "GGUF parsed: \n$metadata")
                withContext(Dispatchers.Main) {
                    setStatus(metadata.toString().take(160))
                }

                // Ensure the model file is available
                val modelName = metadata.filename() + FILE_EXTENSION_GGUF
                contentResolver.openInputStream(uri)?.use { input ->
                    ensureModelFile(modelName, input)
                }?.let { modelFile ->
                    loadModel(modelName, modelFile)

                    withContext(Dispatchers.Main) {
                        markModelReady(modelName)
                    }
                }
            }
        }
    }

    /**
     * Prepare the model file within app's private storage
     */
    private suspend fun ensureModelFile(modelName: String, input: InputStream) =
        withContext(Dispatchers.IO) {
            File(ensureModelsDirectory(), modelName).also { file ->
                // Copy the file into local storage if not yet done
                if (!file.exists()) {
                    Log.i(TAG, "Start copying file to $modelName")
                    withContext(Dispatchers.Main) {
                        userInputEt.hint = "Copying file..."
                    }

                    FileOutputStream(file).use { input.copyTo(it) }
                    Log.i(TAG, "Finished copying file to $modelName")
                } else {
                    Log.i(TAG, "File already exists $modelName")
                }
            }
        }

    /**
     * Automatically load the bundled post-trained model if it has been provisioned
     * into the app's private model directory.
     */
    private suspend fun autoLoadPostTrainedModel() {
        val modelFile = File(ensureModelsDirectory(), POST_TRAINED_MODEL_FILENAME)
        if (!modelFile.exists()) {
            withContext(Dispatchers.Main) {
                setStatus("Post-trained model not provisioned yet")
                userInputEt.hint = "Select a GGUF file to begin"
            }
            return
        }

        try {
            withContext(Dispatchers.Main) {
                userActionFab.isEnabled = false
                userInputEt.hint = "Loading post-trained model..."
                setStatus("Loading $POST_TRAINED_MODEL_FILENAME…")
            }
            loadModel(POST_TRAINED_MODEL_FILENAME, modelFile)
            engine.setSystemPrompt(systemPromptWithRules())
            withContext(Dispatchers.Main) {
                markModelReady("Qwen3.5 ORPO-v4")
            }
            withContext(Dispatchers.Main) {
                // Greet only a blank slate: a restored history already
                // has messages, and re-adding here is what produced the
                // stacked duplicate welcome bubbles after every restart.
                if (messages.isEmpty()) {
                    messages.add(
                        Message(
                            id = UUID.randomUUID().toString(),
                            content = "Hi! I'm running your post-trained Qwen3.5 ORPO-v4 model on-device. " +
                                "I can think out loud and search the web when needed. Ask me anything!",
                            isUser = false,
                        )
                    )
                    persistMessages()
                    messageAdapter.notifyItemInserted(messages.size - 1)
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load post-trained model", e)
            withContext(Dispatchers.Main) {
                setStatus("Post-trained model failed to load")
                userInputEt.hint = "Select a GGUF file to begin"
                userActionFab.isEnabled = true
            }
        }
    }

    /**
     * Load the model file from the app private storage
     */
    private suspend fun loadModel(modelName: String, modelFile: File) =
        withContext(Dispatchers.IO) {
            Log.i(TAG, "Loading model $modelName")
            withContext(Dispatchers.Main) {
                userInputEt.hint = "Loading model..."
            }
            engine.loadModel(modelFile.path)
        }

    /**
     * Copies the GGUF bundled in the APK into app-private storage on the
     * very first launch, so a plain phone install needs no adb staging.
     */
    private suspend fun ensureBundledModelExtracted() {
        val outDir = ensureModelsDirectory()
        val out = File(outDir, POST_TRAINED_MODEL_FILENAME)
        if (out.exists() && out.length() > 0) {
            return  // Already staged (or extracted); temp+rename guaranteed completeness.
        }
        if (!outDir.exists()) outDir.mkdirs()
        val tmp = File(outDir, POST_TRAINED_MODEL_FILENAME + ".part")
        tmp.delete()
        val setStatusText: suspend (String) -> Unit = { t: String ->
            withContext(Dispatchers.Main) { setStatus(t) }
        }
        setStatusText("● Extracting model (first launch)…")
        Log.i(TAG, "Extracting bundled model to ${out.path}")
        try {
            assets.openFd("gguf/$POST_TRAINED_MODEL_FILENAME").use { afd ->
                afd.createInputStream().use { input ->
                    tmp.outputStream().use { fos ->
                        val total = afd.length
                        val buf = ByteArray(1 shl 22) // 4 MiB
                        var copied = 0L
                        var nextReport = 512L shl 20
                        while (copied < total) {
                            // AssetInputStream can return alignment padding
                            // past afd.length; clamp every read to remaining
                            // so the clone ends at exactly the asset bytes.
                            val want = minOf(buf.size.toLong(), total - copied).toInt()
                            val n = input.read(buf, 0, want)
                            if (n <= 0) break
                            fos.write(buf, 0, n)
                            copied += n
                            if (copied >= nextReport) {
                                nextReport += 512L shl 20
                                setStatusText(
                                    "● Extracting model ${copied * 100 / maxOf(total, 1)}%…",
                                )
                            }
                        }
                        fos.flush()
                        if (copied < total) {
                            throw IllegalStateException(
                                "Extraction incomplete: $copied / $total bytes",
                            )
                        }
                    }
                }
            }
            if (!tmp.renameTo(out)) {
                throw IllegalStateException("Could not finalize model extraction")
            }
            Log.i(TAG, "Bundled model extracted: ${out.length()} bytes")
            setStatusText("● Model ready to load…")
        } finally {
            tmp.delete()
        }
    }

    /**
     * Persist the chat so a process restart (crash/update) never wipes it.
     */
    private fun persistMessages() {
        try {
            val arr = org.json.JSONArray()
            messages.takeLast(100).forEach { m ->
                arr.put(
                    org.json.JSONObject()
                        .put("id", m.id)
                        .put("content", m.content)
                        .put("isUser", m.isUser)
                        .put("thinking", m.thinking ?: "")
                        .put("toolLabel", m.toolLabel ?: "")
                )
            }
            File(filesDir, CHAT_HISTORY_FILE).writeText(arr.toString())
        } catch (e: Exception) {
            Log.w(TAG, "Chat persist failed", e)
        }
    }

    private fun restoreMessages(): Boolean {
        return try {
            val f = File(filesDir, CHAT_HISTORY_FILE)
            if (!f.exists()) return false
            val arr = org.json.JSONArray(f.readText())
            messages.clear()
            for (i in 0 until arr.length()) {
                val o = arr.getJSONObject(i)
                messages.add(
                    Message(
                        id = o.getString("id"),
                        content = o.getString("content"),
                        isUser = o.getBoolean("isUser"),
                        thinking = o.optString("thinking").takeIf { it.isNotEmpty() },
                        thinkingExpanded = false,
                        toolLabel = o.optString("toolLabel").takeIf { it.isNotEmpty() }
                    )
                )
            }
            messages.isNotEmpty()
        } catch (e: Exception) {
            Log.w(TAG, "Chat restore failed", e)
            false
        }
    }

    private fun setStatus(text: String) {
        statusTv.text = text
    }

    private fun applySavedTheme() {
        AppCompatDelegate.setDefaultNightMode(
            if (isDarkMode()) AppCompatDelegate.MODE_NIGHT_YES
            else AppCompatDelegate.MODE_NIGHT_NO
        )
    }

    private fun isDarkMode(): Boolean {
        return getSharedPreferences(PREFS_NAME, MODE_PRIVATE)
            .getBoolean(KEY_DARK_MODE, false)
    }

    private fun updateThemeToggleIcon() {
        themeToggleBtn.text = if (isDarkMode()) "☀️" else "🌙"
    }

    /** Shared "model is ready" UI state (fresh load or activity recreation). */
    private fun markModelReady(modelName: String) {
        isModelReady = true
        setStatus("● $modelName ready")
        statusTv.setTextColor(getColor(R.color.status_ready))
        userInputEt.hint = "Message Qwen…"
        userInputEt.isEnabled = true
        userActionFab.setImageResource(R.drawable.outline_send_24)
        userActionFab.isEnabled = true
    }

    /** Reads the hidden rules file shipped in assets (null when absent). */
    private fun loadRulesAsset(): String? = try {
        assets.open(RULES_ASSET).bufferedReader().use { it.readText() }
            .takeIf { it.isNotBlank() }
    } catch (_: Exception) {
        null
    }

    /** System prompt with the rules file appended when it exists. */
    private fun systemPromptWithRules(): String {
        val rules = loadRulesAsset() ?: return SYSTEM_PROMPT
        return "$SYSTEM_PROMPT\n\n$rules"
    }

    /** True when a "final answer" is really leftover tool-call markup or
     * degenerate per-character junk; such output must trigger a retry. */
    private fun junkyAnswer(a: String): Boolean {
        val t = a.replace(" ", "").replace("\n", "").lowercase()
        if (t.isEmpty()) return true
        if ("tool_call" in t || "toolresponse" in t || "function=" in t) return true
        if ("web_search" in t || "search" in t && "query" in t) return true
        // per-character degenerate form: mostly separators
        val printable = t.count { it.isLetterOrDigit() }
        return printable * 2 < t.length
    }

    /**
     * Greeting/thanks/identity/small-talk inputs are not factual claims; the
     * fact-check fine-tune goes off the rails on them (invents a claim and
     * searches for it), so they skip the tool loop entirely.
     */
    private fun isSmallTalk(msg: String): Boolean {
        val m = msg.trim().lowercase()
        // trailing filler words tolerated: "hi there", "hey bot"
        return m.matches(Regex(
            "((hi|hello|heya|hiya|yo|sup|greetings|" +
                "good\\s*(morning|afternoon|evening))" +
                "(\\s*(there|bot|qwen|assistant|friend|all))*" +
                "|hey(\\s*(there|bot|qwen|assistant|friend|all))*)" +
                "(\\s*[!.?])*" +
                "|(thanks|thank\\s*you|ty)(\\s*(a\\s*lot|so\\s*much|bud|bro|assistant))?(\\s*[!.?])*" +
                "|(bye|goodbye|good\\s*night|see\\s*(ya|you))(\\s*(now|later|for\\s*now))?(\\s*[!.?])*" +
                "|(how\\s*(are|r)\\s*(you|u))(\\s*(doing|doing\\s*today|today))?(\\s*[!.?])*" +
                "|what'?s\\s*up(\\s*(doc|man|bro))?(\\s*[!.?])*" +
                "|(who|what)\\s*(are|r)\\s*(you|u)(\\s*(exactly|again))?(\\s*[!.?])*" +
                "|tell\\s*me\\s*about\\s*yours?elf(\\s*[!.?])*" +
                "|what\\s*can\\s*you\\s*do(\\s*for\\s*me)?(\\s*[!.?])*",
            RegexOption.IGNORE_CASE
        ))
    }

    /**
     * Validate and send the user message into [InferenceEngine], handling the
     * model's think + web_search tool loop and rendering a clean chat answer.
     */
    private fun handleUserInput() {
        userInputEt.text.toString().also { userMsg ->
            if (userMsg.isEmpty()) {
                Toast.makeText(this, "Input message is empty!", Toast.LENGTH_SHORT).show()
            } else {
                // The keyboard stays up only if the user is still typing.
                val imm = getSystemService(INPUT_METHOD_SERVICE) as InputMethodManager
                imm.hideSoftInputFromWindow(userInputEt.windowToken, 0)
                // The composer stays editable so the user can type the
                // next message while this turn runs; only sending is
                // disabled until finishTurn().
                userInputEt.text = null
                userActionFab.isEnabled = false
                setStatus("● Generating answer…")

                // Update message states
                messages.add(Message(UUID.randomUUID().toString(), userMsg, true))
                val assistantId = UUID.randomUUID().toString()
                messages.add(Message(assistantId, "Thinking…", false))
                persistMessages()
                messageAdapter.notifyItemInserted(messages.size - 1)
                messagesRv.scrollToPosition(messages.size - 1)

                generationJob = lifecycleScope.launch(Dispatchers.Default) {
                    try {
                        runAssistantTurn(assistantId, userMsg)
                    } catch (e: Exception) {
                        Log.e(TAG, "Generation failed", e)
                        withContext(NonCancellable + Dispatchers.Main) {
                            val note = if (e is CancellationException) {
                                "Stopped."
                            } else {
                                "Sorry, something went wrong: ${e.message}"
                            }
                            updateAssistant(assistantId, content = note)
                            finishTurn()
                        }
                    }
                }
            }
        }
    }

    /**
     * Runs one assistant turn: generate -> optionally execute web_search tool
     * call(s) -> generate final answer. Raw tool_call markup is never shown.
     */
    private suspend fun runAssistantTurn(assistantId: String, userMsg: String) {
        var followUp: String? = null
        var toolRounds = 0
        var webDown = false
        var thinking: String? = null
        var toolLabel: String? = null
        var lastWebResults: String? = null

        // First generation: the user's message.
        val smallTalk = isSmallTalk(userMsg)
        var raw = collectGeneration(userMsg, predictLength = 1024)
        Log.i(TAG, "First raw (first 1500 chars): ${raw.take(1500)}")
        var parsed = ChatParser.parse(raw)
        thinking = parsed.thinking ?: thinking

        // Tool loop: resolve web_search calls, then ask for the final answer.
        while (!smallTalk && parsed.toolCall != null && toolRounds < MAX_TOOL_ROUNDS) {
            val tool = parsed.toolCall!!
            if (tool.name != "web_search" || tool.query.isBlank()) break
            // The model's rewritten queries proved unreliable on-device
            // (garbled/echo/drift); the user's own message IS the query.
            val query = userMsg.trim().trimEnd('.', '?', '!')
            toolRounds++
            toolLabel = "Web search used for: \"${query.take(120)}\""
            withContext(Dispatchers.Main) {
                updateAssistant(assistantId, "Thinking…", thinking, toolLabel)
            }
            Log.i(TAG, "Executing web_search: $query")
            val result = WebSearch.search(query)
            lastWebResults = result
            // Detect tool outage: WebSearch returns failure strings
            // instead of throwing; tell the model so it can fall back.
            webDown = result.startsWith("Web search failed") ||
                result.startsWith("No web results found")
            followUp = if (webDown) {
                " \n$result\n \n" +
                    "The user's original message was exactly: \"$userMsg\". " +
                    "The web search tool isn't available right now. If " +
                    "you know the answer from your own knowledge, answer " +
                    "in 2-4 plain sentences. If you do not know it, reply " +
                    "with exactly this sentence and nothing else: " +
                    "I don't know the answer, and the web search tool " +
                    "isn't available right now."
            } else {
                " \n$result\n \n" +
                    "The user's original message was exactly: \"$userMsg\". " +
                    "Answer THAT message using this result. If " +
                    "it was a question, start with the direct answer and give " +
                    "2-4 plain sentences (entity first, then relevant " +
                    "details). If it was a claim, open with exactly one " +
                    "verdict sentence plus the evidence. Never narrate, " +
                    "never emit another tool_call."
            }
            raw = collectGeneration(followUp, predictLength = 1024)
            Log.i(TAG, "Follow-up raw (first 1500 chars): ${raw.take(1500)}")
            parsed = ChatParser.parse(raw)
            if (parsed.thinking != null) thinking = parsed.thinking
        }

        // Unusable answer (blank, malformed markup, or another tool call
        // after the round cap): ask once more for a direct reply. The
        // fetched web result travels along so the model can answer
        // without calling the tool again.
        if ((parsed.answer.isBlank() || junkyAnswer(parsed.answer)) && !smallTalk) {
            Log.i(TAG, "Unusable answer; retrying with no-tools instruction")
            raw = collectGeneration(
                buildString {
                    append(" (no tools are available right now). Reply directly ")
                    append("in 2-4 plain sentences using your own knowledge")
                    lastWebResults?.let {
                        append(". Web result to use instead of any tool call:\n")
                        append(it)
                    }
                    append(". Do not emit any tool_call. If you do not know ")
                    append("the answer, reply exactly: I don't know the answer, ")
                    append("and the web search tool isn't available right now.")
                },
                predictLength = 1024,
            )
            Log.i(TAG, "Retry raw (first 1500 chars): ${raw.take(1500)}")
            parsed = ChatParser.parse(raw)
            if (parsed.thinking != null) thinking = parsed.thinking
        }

        val answer = parsed.answer.ifBlank {
            // The model never produced a final answer (kept emitting tool
            // calls): present the fetched web results directly instead.
            when {
                smallTalk -> "Hello! I'm Qwen, your on-device assistant. How can I help?"
                webDown -> "I don't know the answer, and the web search tool isn't available right now."
                lastWebResults != null -> "Here's what I found on the web:\n\n$lastWebResults"
                else -> "I couldn't compose an answer. Try asking again."
            }
        }
        withContext(Dispatchers.Main) {
            updateAssistant(assistantId, answer, thinking, toolLabel)
            finishTurn()
        }
    }

    /** Collects the full streamed generation for one prompt. */
    private suspend fun collectGeneration(prompt: String, predictLength: Int): String {
        val sb = StringBuilder()
        engine.sendUserPrompt(prompt, predictLength)
            .onCompletion { /* handled by caller */ }
            .collect { token -> sb.append(token) }
        return sb.toString()
    }

    private fun updateAssistant(
        assistantId: String,
        content: String,
        thinking: String? = null,
        toolLabel: String? = null,
    ) {
        val index = messages.indexOfFirst { it.id == assistantId }
        if (index < 0) return
        val current = messages[index]
        messages[index] = current.copy(
            content = content,
            thinking = thinking ?: current.thinking,
            toolLabel = toolLabel ?: current.toolLabel,
        )
        messageAdapter.notifyItemChanged(index)
        messagesRv.scrollToPosition(index)
        persistMessages()
    }

    private fun confirmNewChat() {
        if (messages.isEmpty()) return
        androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("New chat")
            .setMessage("Clear the current conversation?")
            .setPositiveButton("Clear") { _, _ ->
                generationJob?.cancel()
                messages.clear()
                persistMessages()
                messageAdapter.notifyDataSetChanged()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun finishTurn() {
        userInputEt.isEnabled = true
        userActionFab.isEnabled = true
        setStatus("● Qwen3.5 ORPO-v4 ready")
    }

    /**
     * Run a benchmark with the model file
     */
    @Deprecated("This benchmark doesn't accurately indicate GUI performance expected by app developers")
    private suspend fun runBenchmark(modelName: String, modelFile: File) =
        withContext(Dispatchers.Default) {
            Log.i(TAG, "Starts benchmarking $modelName")
            withContext(Dispatchers.Main) {
                userInputEt.hint = "Running benchmark..."
            }
            engine.bench(
                pp = BENCH_PROMPT_PROCESSING_TOKENS,
                tg = BENCH_TOKEN_GENERATION_TOKENS,
                pl = BENCH_SEQUENCE,
                nr = BENCH_REPETITION
            ).let { result ->
                messages.add(Message(UUID.randomUUID().toString(), result, false))
                withContext(Dispatchers.Main) {
                    messageAdapter.notifyItemChanged(messages.size - 1)
                }
            }
        }

    /**
     * Create the `models` directory if not exist.
     */
    private fun ensureModelsDirectory() =
        File(filesDir, DIRECTORY_MODELS).also {
            if (it.exists() && !it.isDirectory) { it.delete() }
            if (!it.exists()) { it.mkdir() }
        }

    override fun onStop() {
        generationJob?.cancel()
        super.onStop()
    }

    override fun onDestroy() {
        // Only tear down the native engine when the user is really leaving;
        // config changes (rotation/theme) must keep the loaded model alive.
        if (isFinishing && this::engine.isInitialized) {
            try {
                engine.destroy()
            } catch (e: Exception) {
                Log.w(TAG, "Engine destroy failed", e)
            }
        }
        super.onDestroy()
    }

    companion object {
        private val TAG = MainActivity::class.java.simpleName

        private const val PREFS_NAME = "qwen_prefs"
        private const val KEY_DARK_MODE = "dark_mode"

        /** Survives activity recreation (e.g. theme toggle) within the process. */
        private val retainedMessages = mutableListOf<Message>()

        private const val DIRECTORY_MODELS = "models"
        private const val RULES_ASSET = "rules.txt"
        private const val CHAT_HISTORY_FILE = "chat_history.json"
        private const val FILE_EXTENSION_GGUF = ".gguf"
        private const val POST_TRAINED_MODEL_FILENAME = "qwen35_orpo_v4_posttrained_q4_0_kai.gguf"
        private const val MAX_TOOL_ROUNDS = 2

        private const val SYSTEM_PROMPT =
            "You are Qwen, a helpful on-device assistant. " +
                "Begin EVERY response with a <tool_call> for web_search per " +
                "reply, and only when the answer needs current or external " +
                "facts. Never emit tool_call markup in your final answer. " +
                "If no search is needed, give the final answer right after " +
                "the think block."

private const val BENCH_PROMPT_PROCESSING_TOKENS = 512
        private const val BENCH_TOKEN_GENERATION_TOKENS = 128
        private const val BENCH_SEQUENCE = 1
        private const val BENCH_REPETITION = 3
    }
}

fun GgufMetadata.filename() = when {
    basic.name != null -> {
        basic.name?.let { name ->
            basic.sizeLabel?.let { size ->
                "$name-$size"
            } ?: name
        }
    }
    architecture?.architecture != null -> {
        architecture?.architecture?.let { arch ->
            basic.uuid?.let { uuid ->
                "$arch-$uuid"
            } ?: "$arch-${System.currentTimeMillis()}"
        }
    }
    else -> {
        "model-${System.currentTimeMillis().toHexString()}"
    }
}
