# SI Document Assistant

A Streamlit application that builds a local hybrid search (vector + keyword) and Question-Answering pipeline over local files and Google Drive documents using Groq LLMs.

## Features
- **Document Support**: PDF, DOCX, TXT, MD (Local upload & Google Drive links).
- **Chunking**: Overlapping character/word chunks with retained metadata (filename, page number).
- **Embeddings & Vector Search**: `all-MiniLM-L6-v2` via SentenceTransformers stored in FAISS.
- **Hybrid Retrieval**: Combines cosine similarity from FAISS with BM25-style keyword matching.
- **RAG via Groq**: Strictly answers based on context and cites sources with page numbers.

## Setup & Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt