package com.example.llama

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView

data class Message(
    val id: String,
    val content: String,
    val isUser: Boolean,
    val thinking: String? = null,
    val thinkingExpanded: Boolean = false,
    val toolLabel: String? = null,
)

class MessageAdapter(
    private val messages: List<Message>,
    private val onThinkingToggle: (String) -> Unit,
) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    companion object {
        private const val VIEW_TYPE_USER = 1
        private const val VIEW_TYPE_ASSISTANT = 2
    }

    override fun getItemViewType(position: Int): Int {
        return if (messages[position].isUser) VIEW_TYPE_USER else VIEW_TYPE_ASSISTANT
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val layoutInflater = LayoutInflater.from(parent.context)
        return if (viewType == VIEW_TYPE_USER) {
            val view = layoutInflater.inflate(R.layout.item_message_user, parent, false)
            UserMessageViewHolder(view)
        } else {
            val view = layoutInflater.inflate(R.layout.item_message_assistant, parent, false)
            AssistantMessageViewHolder(view)
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val message = messages[position]
        if (holder is UserMessageViewHolder) {
            holder.itemView.findViewById<TextView>(R.id.msg_content).text = message.content
        } else if (holder is AssistantMessageViewHolder) {
            val content = holder.itemView.findViewById<TextView>(R.id.msg_content)
            content.text = message.content.ifEmpty { "…" }

            val thinkingCard =
                holder.itemView.findViewById<LinearLayout>(R.id.thinking_card)
            val thinkingHeader =
                holder.itemView.findViewById<TextView>(R.id.thinking_header)
            val thinkingBody =
                holder.itemView.findViewById<TextView>(R.id.thinking_body)
            if (!message.thinking.isNullOrBlank()) {
                thinkingCard.visibility = View.VISIBLE
                thinkingBody.text = message.thinking
                if (message.thinkingExpanded) {
                    thinkingBody.maxLines = Int.MAX_VALUE
                    thinkingHeader.text = "Thought (tap to collapse)"
                } else {
                    thinkingBody.maxLines = 3
                    thinkingHeader.text = "Thought (tap to expand)"
                }
                thinkingCard.setOnClickListener { onThinkingToggle(message.id) }
            } else {
                thinkingCard.visibility = View.GONE
                thinkingCard.setOnClickListener(null)
            }

            val toolCard = holder.itemView.findViewById<LinearLayout>(R.id.tool_card)
            val toolText = holder.itemView.findViewById<TextView>(R.id.tool_text)
            if (!message.toolLabel.isNullOrBlank()) {
                toolCard.visibility = View.VISIBLE
                toolText.text = message.toolLabel
            } else {
                toolCard.visibility = View.GONE
            }
        }
    }

    override fun getItemCount(): Int = messages.size

    class UserMessageViewHolder(view: View) : RecyclerView.ViewHolder(view)
    class AssistantMessageViewHolder(view: View) : RecyclerView.ViewHolder(view)
}
