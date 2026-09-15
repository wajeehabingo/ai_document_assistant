import os
import re
import tempfile
import numpy as np
import streamlit as st
import docx
import pypdf
import gdown
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ------------------------------------------------------------------------------
# 1. INITIALIZATION & SESSION STATE
# ------------------------------------------------------------------------------
st.set_page_config(page_title="SI Document Assistant", layout="wide")

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "processed" not in st.session_state:
    st.session_state.processed = False

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()

# ------------------------------------------------------------------------------
# 2. DOCUMENT EXTRACTION FUNCTIONS
# ------------------------------------------------------------------------------
def extract_pdf(file_path, filename):
    docs = []
    reader = pypdf.PdfReader(file_path)
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append({"filename": filename, "page": idx + 1, "text": text})
    return docs

def extract_docx(file_path, filename):
    doc = docx.Document(file_path)
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return [{"filename": filename, "page": "N/A", "text": full_text}]

def extract_txt_md(file_path, filename):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{"filename": filename, "page": "N/A", "text": text}]

def extract_document(file_path, filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_pdf(file_path, filename)
    elif ext == ".docx":
        return extract_docx(file_path, filename)
    elif ext in [".txt", ".md"]:
        return extract_txt_md(file_path, filename)
    return []

# ------------------------------------------------------------------------------
# 3. TEXT CHUNKING
# ------------------------------------------------------------------------------
def chunk_documents(docs, chunk_size=500, overlap=100):
    chunks = []
    for doc in docs:
        text = doc["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({
                "filename": doc["filename"],
                "page": doc["page"],
                "text": chunk_text
            })
            start += (chunk_size - overlap)
    return chunks

# ------------------------------------------------------------------------------
# 4. EMBEDDINGS & FAISS INDEXING
# ------------------------------------------------------------------------------
def create_vector_store(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    
    dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    return index, embeddings

# ------------------------------------------------------------------------------
# 5. HYBRID SEARCH (VECTOR + KEYWORD)
# ------------------------------------------------------------------------------
def keyword_search(query, chunks):
    keywords = re.findall(r"\w+", query.lower())
    scores = []
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        score = sum(text_lower.count(kw) for kw in keywords)
        scores.append(score)
    
    max_score = max(scores) if max(scores) > 0 else 1
    return [s / max_score for s in scores]

def hybrid_search(query, chunks, faiss_index, top_k=4, alpha=0.6):
    # Vector Search via FAISS
    query_emb = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    vector_scores, indices = faiss_index.search(query_emb, len(chunks))
    
    # Map vector scores back to document indices
    v_scores = np.zeros(len(chunks))
    for score, idx in zip(vector_scores[0], indices[0]):
        if idx != -1:
            v_scores[idx] = score

    # Keyword Search
    kw_scores = keyword_search(query, chunks)

    # Hybrid Score Calculation
    combined_scores = [
        alpha * v_scores[i] + (1 - alpha) * kw_scores[i] 
        for i in range(len(chunks))
    ]

    # Return Top K results
    ranked_indices = np.argsort(combined_scores)[::-1][:top_k]
    return [chunks[i] for i in ranked_indices]

# ------------------------------------------------------------------------------
# 6. STREAMLIT UI & PROCESSING PIPELINE
# ------------------------------------------------------------------------------
st.title("📄 SI Document Assistant")

with st.sidebar:
    st.header("Document Ingestion")
    
    # Upload local files
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )
    
    # Link Google Drive
    gdrive_link = st.text_input("Or paste Google Drive link (File/Folder)")
    
    if st.button("Process Documents"):
        all_raw_docs = []
        
        # 1. Process local file uploads
        with st.spinner("Processing local files..."):
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    extracted = extract_document(tmp_path, uploaded_file.name)
                    all_raw_docs.extend(extracted)
                    os.remove(tmp_path)

        # 2. Process Google Drive link
        with st.spinner("Processing Google Drive link..."):
            if gdrive_link:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    try:
                        gdown.download_folder(url=gdrive_link, output=tmp_dir, quiet=True)
                    except Exception:
                        gdown.download(url=gdrive_link, output=os.path.join(tmp_dir, "gdoc_file"), quiet=True)
                    
                    for root, _, files in os.walk(tmp_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            extracted = extract_document(file_path, file)
                            all_raw_docs.extend(extracted)

        # 3. Text Chunking, Embeddings, and FAISS indexing
        if all_raw_docs:
            chunks = chunk_documents(all_raw_docs)
            st.session_state.chunks = chunks
            
            with st.spinner("Generating embeddings & FAISS index..."):
                index, _ = create_vector_store(chunks)
                st.session_state.faiss_index = index
                st.session_state.processed = True
                
            st.success(f"Successfully processed {len(all_raw_docs)} document entries into {len(chunks)} chunks!")
        else:
            st.error("No valid text extracted from the provided inputs.")

# ------------------------------------------------------------------------------
# 7. QUESTION ANSWERING INTERFACE
# ------------------------------------------------------------------------------
if st.session_state.processed:
    st.write(f"**Total chunks ready for search:** `{len(st.session_state.chunks)}`")
    query = st.text_input("Ask a question about your documents:")

    if query:
        # Hybrid retrieval
        retrieved_chunks = hybrid_search(
            query, 
            st.session_state.chunks, 
            st.session_state.faiss_index
        )
        
        context_str = "\n\n".join(
            [f"[Source: {c['filename']}, Page: {c['page']}]\n{c['text']}" for c in retrieved_chunks]
        )

        # Call Groq Model via API Key stored in Streamlit Secrets
        if "GROQ_API_KEY" not in st.secrets:
            st.error("GROQ_API_KEY is not set in `.streamlit/secrets.toml`!")
        else:
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            
            prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the provided context below.
If the answer cannot be found in the context, explicitly state "Information not available in the provided context."

Context:
{context_str}

Question:
{query}
"""

            with st.spinner("Generating answer..."):
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="openai/gpt-oss-120b"
                )
                answer = chat_completion.choices[0].message.content

            st.markdown("### Answer")
            st.write(answer)

            # Display Retrieved Sources Below Answer
            st.markdown("---")
            st.markdown("### Retrieved Sources")
            for i, chunk in enumerate(retrieved_chunks, 1):
                with st.expander(f"Source {i}: {chunk['filename']} (Page {chunk['page']})"):
                    st.write(chunk["text"])
