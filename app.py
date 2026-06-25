import gradio as gr
from src.rag_pipeline import answer_question, retrieve

# ── Simple LLM using HuggingFace pipeline ──
# We use a small model that runs on CPU
from transformers import pipeline

print("Loading LLM... please wait")
llm_pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",  # small, CPU-friendly, no GPU needed
    max_new_tokens=300
)
print("LLM loaded ✅")

def llm_fn(prompt: str) -> str:
    result = llm_pipe(prompt, max_new_tokens=300)
    return result[0]['generated_text']

def chat(question):
    if not question.strip():
        return "Please enter a question.", ""
    
    answer, sources = answer_question(question, llm_fn, k=5)
    
    source_text = "\n\n---\n### 📄 Sources Used:\n"
    for i, s in enumerate(sources, 1):
        source_text += f"\n**[{i}] {s['product_category']}** (Complaint {s['complaint_id']})\n"
        source_text += f"> {s['text'][:300]}...\n"
    
    return answer, source_text

# ── Gradio Interface ──
with gr.Blocks(title="CrediTrust Complaint Analyst") as demo:
    gr.Markdown("# 🏦 CrediTrust Financial — Complaint Analyst Chatbot")
    gr.Markdown("Ask any question about customer complaints across Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.")
    
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(
                label="Your Question",
                placeholder="e.g. What are the most common credit card complaints?",
                lines=3
            )
            with gr.Row():
                ask_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear")
    
    with gr.Row():
        with gr.Column():
            answer_box = gr.Textbox(label="Answer", lines=6)
        with gr.Column():
            sources_box = gr.Markdown(label="Sources")
    
    ask_btn.click(
        fn=chat,
        inputs=question,
        outputs=[answer_box, sources_box]
    )
    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[question, answer_box, sources_box]
    )
if __name__ == "__main__":
    demo.launch() 