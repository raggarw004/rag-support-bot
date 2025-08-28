import os
import gradio as gr
from rag_chat import RAGChatbot
from config import DEFAULT_OLLAMA_MODEL

bot = None

def init_bot(model_name):
    global bot
    bot = RAGChatbot(model_name=model_name)
    return f"Model loaded: {model_name}. Vector store ready."

def chat_fn(history, message):
    if bot is None:
        return history + [[message, "Error: bot not ready"]]
    answer, hits = bot.generate(message)
    # Append brief citations
    cites = "  \n".join({f"[source: {os.path.basename(h['source'])}]" for h in hits})
    answer = f"{answer}\n\n---\n{cites}"
    return history + [[message, answer]]

with gr.Blocks(title="RAG Customer Support Bot") as demo:
    gr.Markdown("# 🛟 RAG Customer Support Bot (Local & Free)")
    model_name = gr.Textbox(value=DEFAULT_OLLAMA_MODEL, label="Ollama model", interactive=True)
    status = gr.Markdown()
    init_btn = gr.Button("Initialize")
    init_btn.click(fn=init_bot, inputs=model_name, outputs=status)

    chat = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Ask a question", placeholder="e.g., How do I reset my password?")
    send = gr.Button("Send")

    def _reply(user_msg, hist):
        return chat_fn(hist, user_msg), ""

    send.click(_reply, inputs=[msg, chat], outputs=[chat, msg])
    msg.submit(_reply, inputs=[msg, chat], outputs=[chat, msg])

if __name__ == "__main__":
    demo.launch()
