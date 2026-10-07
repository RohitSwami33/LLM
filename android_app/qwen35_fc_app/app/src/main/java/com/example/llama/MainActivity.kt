package com.example.llama

import android.net.Uri
import android.os.Bundle
import android.util.Log
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

        newChatBtn.setOnClickListener {
            generationJob?.cancel()
            messages.clear()
            messageAdapter.notifyDataSetChanged()
        }

        // Arm AI Chat initialization
        lifecycleScope.launch(Dispatchers.Default) {
            engine = AiChat.getInferenceEngine(applicationContext)
            if (engine.state.value is InferenceEngine.State.ModelReady &&
                File(ensureModelsDirectory(), POST_TRAINED_MODEL_FILENAME).exists()
            ) {
                // Activity recreated (e.g. theme change): model still loaded.
                withContext(Dispatchers.Main) { markModelReady("Qwen3.5 ORPO-v4") }
            } else {
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
            engine.setSystemPrompt(SYSTEM_PROMPT)
            withContext(Dispatchers.Main) {
                markModelReady("Qwen3.5 ORPO-v4")
            }
            withContext(Dispatchers.Main) {
                messages.add(
                    Message(
                        id = UUID.randomUUID().toString(),
                        content = "Hi! I'm running your post-trained Qwen3.5 ORPO-v4 model on-device. " +
                            "I can think out loud and search the web when needed. Ask me anything!",
                        isUser = false,
                    )
                )
                messageAdapter.notifyItemInserted(messages.size - 1)
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

    /**
     * Validate and send the user message into [InferenceEngine], handling the
     * model's think + web_search tool loop and rendering a clean chat answer.
     */
    private fun handleUserInput() {
        userInputEt.text.toString().also { userMsg ->
            if (userMsg.isEmpty()) {
                Toast.makeText(this, "Input message is empty!", Toast.LENGTH_SHORT).show()
            } else {
                userInputEt.text = null
                userInputEt.isEnabled = false
                userActionFab.isEnabled = false

                // Update message states
                messages.add(Message(UUID.randomUUID().toString(), userMsg, true))
                val assistantId = UUID.randomUUID().toString()
                messages.add(Message(assistantId, "Thinking…", false))
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
        var thinking: String? = null
        var toolLabel: String? = null
        var lastWebResults: String? = null

        // First generation: the user's message.
        var raw = collectGeneration(userMsg, predictLength = 1024)
        Log.i(TAG, "First raw (first 1500 chars): ${raw.take(1500)}")
        var parsed = ChatParser.parse(raw)
        thinking = parsed.thinking ?: thinking

        // Tool loop: resolve web_search calls, then ask for the final answer.
        while (parsed.toolCall != null && toolRounds < MAX_TOOL_ROUNDS) {
            val tool = parsed.toolCall!!
            if (tool.name != "web_search" || tool.query.isBlank()) break
            toolRounds++
            toolLabel = "Web search used for: \"${tool.query.take(120)}\""
            withContext(Dispatchers.Main) {
                updateAssistant(assistantId, "Thinking…", thinking, toolLabel)
            }
            Log.i(TAG, "Executing web_search: ${tool.query}")
            val result = WebSearch.search(tool.query)
            lastWebResults = result
            followUp = "<tool_response>\n$result\n</tool_response>\n" +
                "The tool result is above. Your reply must contain ONLY the " +
                "bare final answer. Never announce, narrate, or explain what " +
                "you are about to do, and never emit another tool_call."
            raw = collectGeneration(followUp, predictLength = 1024)
            Log.i(TAG, "Follow-up raw (first 1500 chars): ${raw.take(1500)}")
            parsed = ChatParser.parse(raw)
            if (parsed.thinking != null) thinking = parsed.thinking
        }

        val answer = parsed.answer.ifBlank {
            // The model never produced a final answer (kept emitting tool
            // calls): present the fetched web results directly instead.
            lastWebResults?.let { "Here's what I found on the web:\n\n$it" }
                ?: "I couldn't compose an answer. Try asking again."
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
    }

    private fun finishTurn() {
        userInputEt.isEnabled = true
        userActionFab.isEnabled = true
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
        engine.destroy()
        super.onDestroy()
    }

    companion object {
        private val TAG = MainActivity::class.java.simpleName

        private const val PREFS_NAME = "qwen_prefs"
        private const val KEY_DARK_MODE = "dark_mode"

        /** Survives activity recreation (e.g. theme toggle) within the process. */
        private val retainedMessages = mutableListOf<Message>()

        private const val DIRECTORY_MODELS = "models"
        private const val FILE_EXTENSION_GGUF = ".gguf"
        private const val POST_TRAINED_MODEL_FILENAME = "qwen35_orpo_v4_posttrained_q4_k_m.gguf"
        private const val MAX_TOOL_ROUNDS = 2

        private const val SYSTEM_PROMPT =
            "You are Qwen, a helpful on-device assistant. " +
                "Begin EVERY response with a <think>...</think> block holding " +
                "your step-by-step reasoning. Then decide: if you need current " +
                "or external facts, emit exactly one <tool_call> for web_search " +
                "with a short SELF-CONTAINED query (include all needed context, " +
                "never copy earlier questions), then wait for the tool result. " +
                "After receiving <tool_response>, reply with ONLY the final " +
                "answer in plain sentences: no tool_call markup, no 'Final " +
                "Answer:' prefix. If no search is needed, give the final answer " +
                "right after the think block."

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
