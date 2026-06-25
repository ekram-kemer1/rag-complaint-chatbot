import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Load vector store ──
index = faiss.read_index("vector_store/faiss_index.bin")
with open("vector_store/chunk_metadata.pkl", "rb") as f:
    chunks = pickle.load(f)

# ── Load embedding model ──
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve(query: str, k: int = 5):
    """Retrieve top-k most relevant chunks for a query."""
    query_vec = embedder.encode([query]).astype('float32')
    distances, indices = index.search(query_vec, k)
    results = []
    for i, idx in enumerate(indices[0]):
        chunk = chunks[idx].copy()
        chunk['score'] = float(distances[0][i])
        results.append(chunk)
    return results

# ── Prompt template ──
PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust Financial.
Your job is to answer questions about customer complaints.
Use ONLY the context below to answer.
If the context does not contain the answer, say:
"I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Answer:
"""

def build_prompt(question: str, retrieved_chunks: list) -> str:
    context = "\n\n".join([
        f"[{c['product_category']}] {c['text']}"
        for c in retrieved_chunks
    ])
    return PROMPT_TEMPLATE.format(context=context, question=question)

def answer_question(question: str, llm_fn, k: int = 5):
    """Full RAG pipeline: retrieve → prompt → generate."""
    retrieved = retrieve(question, k=k)
    prompt = build_prompt(question, retrieved)
    answer = llm_fn(prompt)
    return answer, retrieved
  
