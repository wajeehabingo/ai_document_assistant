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
from openai import OpenAI

# ------------------------------------------------------------------------------
# 1. INITIALIZATION, PAGE CONFIG & CUSTOM CSS THEMING
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Bluish-Greenish (Teal/Emerald/Dark) Styling
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #0b131e;
        color: #e2e8f0;
    }
    
    /* Headers & Branding */
    h1, h2, h3, h4 {
        color: #2dd4bf !important;
        font-weight: 700 !important;
    }
    
    /* Custom Gradient Card Header */
    .header-card {
        background: linear-gradient(135deg, #0f766e 0%, #1e3a8a 100%);
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border: 1px solid #14b8a6;
    }
    .header-card h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: 2.2rem;
    }
    .header-card p {
        color: #99f6e4;
        margin-top: 5px;
        font-size: 1.05rem;
    }

    /* Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #111827;
        padding: 8px;
        border-radius: 10px;
        border: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        padding: 0px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0d9488 !important;
        color: #ffffff !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0d9488 0%, #0f766e 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(13, 148, 136, 0.4);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #14b8a6 0%, #0d9488 100%);
        box-shadow: 0 4px 12px rgba(20, 184, 166, 0.6);
        color: #ffffff;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }

    /* Source Expander Styling */
    .stExpander {
        background-color: #111827;
        border: 1px solid #1f2937 !important;
        border-radius: 8px !important;
        margin-bottom: 10px;
    }
    
    /* Input Fields */
    .stTextInput input {
        background-color: #1e293b !important;
        color: #f8fafc !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    .stTextInput input:focus {
        border-color: #2dd4bf !important;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "docs_summary" not in st.session_state:
    st.session_state.docs_summary = []

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
    query_emb = embedding_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    vector_scores, indices = faiss_index.search(query_emb, len(chunks))
    
    v_scores = np.zeros(len(chunks))
    for score, idx in zip(vector_scores[0], indices[0]):
        if idx != -1:
            v_scores[idx] = score

    kw_scores = keyword_search(query, chunks)

    combined_scores = [
        alpha * v_scores[i] + (1 - alpha) * kw_scores[i] 
        for i in range(len(chunks))
    ]

    ranked_indices = np.argsort(combined_scores)[::-1][:top_k]
    return [chunks[i] for i in ranked_indices]

# ------------------------------------------------------------------------------
# 6. HEADER & SIDEBAR INGESTION PIPELINE
# ------------------------------------------------------------------------------
st.markdown("""
<div class="header-card">
    <h1>🤖 AI Document Assistant</h1>
    <p>Hybrid Semantic Search & Context-Aware QA Engine for PDF, DOCX, TXT, MD & Google Drive</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📂 Data Source Portal")
    st.markdown("---")
    
    uploaded_files = st.file_uploader(
        "Upload local files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )
    
    st.markdown("**OR**")
    gdrive_link = st.text_input("Google Drive Link (Folder/File)")
    
    st.markdown("---")
    if st.button("🚀 Process Documents", use_container_width=True):
        all_raw_docs = []
        file_summary = []
        
        with st.spinner("Processing local files..."):
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name
                    extracted = extract_document(tmp_path, uploaded_file.name)
                    all_raw_docs.extend(extracted)
                    file_summary.append({"name": uploaded_file.name, "source": "Local Upload"})
                    os.remove(tmp_path)

        with st.spinner("Processing Google Drive links..."):
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
                            file_summary.append({"name": file, "source": "Google Drive"})

        if all_raw_docs:
            chunks = chunk_documents(all_raw_docs)
            st.session_state.chunks = chunks
            st.session_state.docs_summary = file_summary
            
            with st.spinner("Embedding and building FAISS vector store..."):
                index, _ = create_vector_store(chunks)
                st.session_state.faiss_index = index
                st.session_state.processed = True
                
            st.success("Documents Ingested Successfully!")
        else:
            st.error("No extractable content found in provided files.")

# ------------------------------------------------------------------------------
# 7. MAIN INTERFACE & TAB NAVIGATION
# ------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["💬 Assistant & Search", "📚 Document Library", "⚙️ System Info"])

with tab1:
    if not st.session_state.processed:
        st.info("👈 Please upload documents or add a Google Drive link in the sidebar to get started.")
    else:
        st.subheader("Ask Questions Over Your Documents")
        query = st.text_input("Enter your query:", placeholder="e.g. What are the key terms in the project agreement?")

        if query:
            retrieved_chunks = hybrid_search(
                query, 
                st.session_state.chunks, 
                st.session_state.faiss_index
            )
            
            context_str = "\n\n".join(
                [f"[Source: {c['filename']}, Page: {c['page']}]\n{c['text']}" for c in retrieved_chunks]
            )

            if "OPENAI_API_KEY" not in st.secrets:
                st.error("Missing `OPENAI_API_KEY` in `.streamlit/secrets.toml` configuration.")
            else:
                client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                
                prompt = f"""You are an expert document assistant. Answer the user's question using ONLY the context provided below.
If the information cannot be found in the context, state "Information not available in the provided context."

Context:
{context_str}

Question:
{query}
"""

                with st.spinner("Generating answer..."):
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model="openai/gpt-oss-120b"
                    )
                    answer = response.choices[0].message.content

                st.markdown("### Answer")
                st.info(answer)

                st.markdown("### 🔍 Retrieved Context Sources")
                for i, chunk in enumerate(retrieved_chunks, 1):
                    with st.expander(f"📍 Source {i}: {chunk['filename']} (Page {chunk['page']})"):
                        st.write(chunk["text"])

with tab2:
    st.subheader("Loaded Documents Summary")
    if st.session_state.processed:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Extracted Chunks", len(st.session_state.chunks))
        with col2:
            st.metric("Total Documents Processed", len(st.session_state.docs_summary))
        
        st.markdown("#### Document Files List")
        for item in st.session_state.docs_summary:
            st.markdown(f"- **{item['name']}** *(Source: {item['source']})*")
    else:
        st.write("No documents loaded yet.")

with tab3:
    st.subheader("Pipeline Configuration & Status")
    st.json({
        "Embedding Model": "SentenceTransformers (all-MiniLM-L6-v2)",
        "Vector Database": "FAISS (IndexFlatIP)",
        "Search Strategy": "Hybrid (Cosine Vector Sim + Keyword Matching)",
        "LLM Provider": "openai/gpt-oss-120b",
        "Ingested Chunks": len(st.session_state.chunks) if st.session_state.processed else 0
    })
