# RAG Complaint Chatbot — CrediTrust Financial

A Retrieval-Augmented Generation (RAG) chatbot that lets internal teams
query CFPB customer complaint data using plain English.

## Project Structure
- `data/` — Raw and processed complaint data
- `notebooks/` — EDA and preprocessing notebooks
- `src/` — Core RAG pipeline modules
- `vector_store/` — Persisted FAISS/ChromaDB index
- `app.py` — Gradio chat interface

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
python app.py
```
