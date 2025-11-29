
import io
import os
import re
import math
import hashlib
import json
import time
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Lazy imports for heavy optional libraries
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pdfplumber
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    pdfplumber = None
    pytesseract = None
    Image = None
    OCR_AVAILABLE = False

try:
    import faiss
    import numpy as np
except Exception:
    faiss = None
    np = None

try:
    import cohere
except Exception:
    cohere = None

import requests

# ----------------------
# Configuration (env)
# ----------------------
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")  # or set directly for local dev (not recommended)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-3.5-turbo-0613")
CHAT_URL = os.getenv("CHAT_URL", "https://openrouter.ai/api/v1/chat/completions")
EMBED_MODEL = os.getenv("EMBED_MODEL", "small")  # Cohere embedding model

# ----------------------
# Page config & intro
# ----------------------
st.set_page_config(page_title="Cohere RAG PDF — Professional", layout="wide")
st.title("📄 Cohere RAG PDF — Q&A")
st.markdown(
    "Upload PDFs, and ask questions. Uses Cohere embeddings + FAISS for retrieval and OpenRouter chat for answers."
)

# ----------------------
# Helper / utilities
# ----------------------
def md5_bytes(b: bytes) -> str:
    h = hashlib.md5()
    h.update(b)
    return h.hexdigest()

def safe_strip(s: Optional[str]) -> str:
    return (s or "").strip()

# ----------------------
# Cohere client (cached)
# ----------------------
@st.cache_resource
def load_cohere_client():
    if not COHERE_API_KEY:
        st.error("Cohere API key not configured. Set COHERE_API_KEY in .env or environment.")
        st.stop()
    if cohere is None:
        st.error("`cohere` package not installed. See requirements.txt")
        st.stop()
    return cohere.Client(COHERE_API_KEY)

# ----------------------
# PDF extraction + OCR fallback
# ----------------------
def extract_pages_from_pdf_bytes(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Returns list of pages: [{"page_no": int, "text": str}, ...]
    Tries PyMuPDF first, then OCR fallback if installed and needed.
    """
    pages = []
    if fitz is None:
        st.warning("PyMuPDF not installed — text extraction may fail.")
        return pages

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for pno, page in enumerate(doc, start=1):
            text = safe_strip(page.get_text("text"))
            pages.append({"page_no": pno, "text": text})
    except Exception as e:
        st.warning(f"PyMuPDF extraction failed: {e}")
        pages = []

    # If extracted text is very small (likely scanned), try OCR if available
    total_chars = sum(len(p.get("text","")) for p in pages)
    if total_chars < 200 and OCR_AVAILABLE:
        ocr_pages = ocr_pdf_bytes(file_bytes)
        if sum(len(p.get("text","")) for p in ocr_pages) > total_chars:
            return ocr_pages

    return pages

def ocr_pdf_bytes(file_bytes: bytes) -> List[Dict[str, Any]]:
    """Perform OCR per page using pdfplumber + pytesseract."""
    pages = []
    if not OCR_AVAILABLE:
        return pages
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for pno, p in enumerate(pdf.pages, start=1):
                pil = p.to_image(resolution=300).original
                text = safe_strip(pytesseract.image_to_string(pil))
                pages.append({"page_no": pno, "text": text})
    except Exception as e:
        st.warning(f"OCR failed: {e}")
    return pages

# ----------------------
# Chunking with provenance
# ----------------------
def chunk_pages_with_provenance(pages: List[Dict[str, Any]],
                                source_filename: str,
                                chunk_size_words: int = 500,
                                overlap_words: int = 50) -> List[Dict[str, Any]]:
    """
    Splits page-level text into sentence-aware chunks with provenance.
    Returns list of dicts: {id, text, file, start_page, end_page}
    """
    sentence_splitter = re.compile(r'(?<=[.!?])\s+')
    sentences_with_page = []
    for p in pages:
        text = p.get("text", "")
        if not text:
            continue
        sentences = sentence_splitter.split(text)
        for s in sentences:
            s = s.strip()
            if s:
                sentences_with_page.append({"text": s, "page": p["page_no"]})

    chunks = []
    cur_words = []
    cur_start_page = None
    cur_end_page = None
    cur_word_count = 0
    chunk_id = 0

    for s in sentences_with_page:
        sent = s["text"]
        pno = s["page"]
        words = sent.split()
        if cur_start_page is None:
            cur_start_page = pno
            cur_end_page = pno
        # if adding this sentence exceeds chunk size, flush previous chunk
        if cur_word_count + len(words) > chunk_size_words and cur_word_count > 0:
            chunks.append({
                "id": chunk_id,
                "text": " ".join(cur_words).strip(),
                "file": source_filename,
                "start_page": cur_start_page,
                "end_page": cur_end_page
            })
            chunk_id += 1
            # carryover last overlap_words
            tail = " ".join(cur_words).split()[-overlap_words:] if overlap_words > 0 else []
            cur_words = tail + words
            cur_word_count = len(cur_words)
            cur_start_page = pno if not tail else cur_start_page
            cur_end_page = pno
        else:
            cur_words.extend(words)
            cur_word_count += len(words)
            cur_end_page = pno

    if cur_words:
        chunks.append({
            "id": chunk_id,
            "text": " ".join(cur_words).strip(),
            "file": source_filename,
            "start_page": cur_start_page or 1,
            "end_page": cur_end_page or 1
        })
    return chunks

# ----------------------
# Embeddings & FAISS helpers
# ----------------------
def embed_texts_with_cohere(texts: List[str], batch_size: int = 128) -> "np.ndarray":
    """
    Returns normalized float32 numpy array of embeddings (shape: [n, dim]).
    """
    if cohere is None:
        raise RuntimeError("cohere package not installed")
    client = load_cohere_client()
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embed(model=EMBED_MODEL, texts=batch)
        # resp.embeddings is expected (Cohere python SDK)
        embs.extend(resp.embeddings)
    arr = np.array(embs, dtype=np.float32)
    if arr.size == 0:
        return arr
    faiss.normalize_L2(arr)
    return arr

def build_faiss_index_from_chunks(chunks_meta: List[Dict[str, Any]]):
    """
    Builds a FAISS IndexFlatIP index from chunk texts and returns (index, chunks_meta)
    """
    texts = [c["text"] for c in chunks_meta]
    if len(texts) == 0:
        raise ValueError("No texts to index")
    embeddings = embed_texts_with_cohere(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, chunks_meta

def retrieve_top_k(index, chunks_meta: List[Dict[str, Any]], question: str, k: int = 4):
    """
    Returns list of dicts: text, file, start_page, end_page, score
    """
    q_emb = embed_texts_with_cohere([question])
    if q_emb.size == 0:
        return []
    scores, ids = index.search(q_emb, k)
    out = []
    for idx, score in zip(ids[0], scores[0]):
        if int(idx) < 0 or int(idx) >= len(chunks_meta):
            continue
        m = chunks_meta[int(idx)]
        out.append({
            "text": m["text"],
            "file": m["file"],
            "start_page": m["start_page"],
            "end_page": m["end_page"],
            "score": float(score)
        })
    return out

# ----------------------
# OpenRouter Chat call
# ----------------------
def call_openrouter_chat(messages: List[Dict[str, str]], timeout: int = 60) -> str:
    """
    messages: list of {"role": "system"/"user"/"assistant", "content": "..."}
    Returns response text or error message.
    """
    if not OPENROUTER_API_KEY:
        st.error("OpenRouter API key not configured. Set OPENROUTER_API_KEY in .env or environment.")
        return "⚠️ Missing API key."
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": CHAT_MODEL, "messages": messages}
    try:
        resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Adapt to common shapes
        if "choices" in data and len(data["choices"]) > 0:
            # OpenRouter-style: choices[].message.content
            if "message" in data["choices"][0]:
                return data["choices"][0]["message"].get("content", "") or ""
            # fallback
            return data["choices"][0].get("text", "") or ""
        if "text" in data:
            return data["text"]
        return json.dumps(data)
    except Exception as e:
        st.error(f"Chat API error: {e}")
        return f"⚠️ Chat API error: {e}"

# ----------------------
# Streamlit UI
# ----------------------
st.sidebar.header("Settings")
chunk_size_words = st.sidebar.number_input("Chunk size (words)", value=500, min_value=200, max_value=2000, step=50)
overlap_words = st.sidebar.number_input("Overlap (words)", value=50, min_value=0, max_value=500, step=10)
top_k = st.sidebar.slider("Top K chunks", 1, 8, 4)
embedding_batch = st.sidebar.number_input("Embedding batch size", value=128, min_value=16, max_value=512, step=16)
show_provenance = st.sidebar.checkbox("Show provenance", value=True)
show_raw = st.sidebar.checkbox("Show raw chunk text", value=False)
st.sidebar.markdown("---")
st.sidebar.info("Do not commit .env with keys. Use GitHub Secrets in production.")

uploaded_files = st.file_uploader("Upload PDF file(s)", type=["pdf"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload one or more PDFs to start. Tip: upload smaller PDF first for a quick test.")
    st.stop()

# Extract text for each file
all_pages_by_file = {}
file_hashes = []
for up in uploaded_files:
    file_bytes = up.read()
    file_hashes.append(md5_bytes(file_bytes + up.name.encode()))
    pages = extract_pages_from_pdf_bytes(file_bytes)
    all_pages_by_file[up.name] = pages

# fingerprint for caching
fingerprint = hashlib.md5("".join(file_hashes).encode()).hexdigest()
index_key = f"faiss_{fingerprint}"
chunks_key = f"chunks_{fingerprint}"

# Build chunks and index if not cached
if index_key in st.session_state and chunks_key in st.session_state:
    index = st.session_state[index_key]
    chunks_meta = st.session_state[chunks_key]
else:
    # Chunk
    chunks_meta = []
    for fname, pages in all_pages_by_file.items():
        chunks_meta.extend(chunk_pages_with_provenance(pages, fname, chunk_size_words, overlap_words))
    if not chunks_meta:
        st.error("No text extracted from PDFs. Enable OCR dependencies or try different files.")
        st.stop()

    # Build vector index
    if faiss is None or np is None:
        st.error("faiss or numpy not installed. See requirements.txt.")
        st.stop()

    with st.spinner("Building embeddings & FAISS index (this may take a moment)..."):
        try:
            index, _ = build_faiss_index_from_chunks(chunks_meta)
            st.session_state[index_key] = index
            st.session_state[chunks_key] = chunks_meta
        except Exception as e:
            st.error(f"Index build error: {e}")
            st.stop()

# Query
st.subheader("Ask a question about the uploaded PDF(s)")
question = st.text_input("Type your question and press Enter")

if not question:
    st.stop()

with st.spinner("Retrieving relevant context..."):
    top_chunks = retrieve_top_k(index, chunks_meta, question, k=top_k)

if not top_chunks:
    st.warning("No relevant context found. Try a broader question.")
    st.stop()

with st.expander("Retrieved context (click to expand)"):
    for i, c in enumerate(top_chunks, start=1):
        cols = st.columns([4, 1])
        with cols[0]:
            st.markdown(f"**Chunk {i}** — score: `{c['score']:.3f}`")
            if show_raw:
                st.write(c["text"])
            else:
                st.write(c["text"][:1200] + ("..." if len(c["text"]) > 1200 else ""))
        with cols[1]:
            if show_provenance:
                st.markdown(f"**File:** {c['file']}  \n**Pages:** {c['start_page']}-{c['end_page']}")

# Compose prompt
system_msg = {
    "role": "system",
    "content": "You are a helpful assistant. Answer the user's question using ONLY the provided excerpts. If the provided text is insufficient, say you don't know."
}
user_context = "\n\n---\n".join([f"Source (pages {c['start_page']}-{c['end_page']} from {c['file']}):\n{c['text']}" for c in top_chunks])
user_msg = {
    "role": "user",
    "content": f"Use the following excerpts to answer the question. If you cannot answer, say you don't know.\n\n{user_context}\n\nQuestion: {question}"
}

with st.spinner("Generating answer via OpenRouter..."):
    answer = call_openrouter_chat([system_msg, user_msg])

st.subheader("Answer")
st.write(answer)

with st.expander("Follow-up question suggestions"):
    fu_prompt = f"Suggest three short, useful follow-up questions a user could ask based on this answer:\n\nAnswer:\n{answer}"
    fu_resp = call_openrouter_chat([{"role": "user", "content": fu_prompt}])
    followups = re.split(r'[\r\n]+', fu_resp)
    followups = [re.sub(r'^\s*\d+[\).]?\s*', '', fu).strip() for fu in followups if fu.strip()]
    for i, fu in enumerate(followups[:5], start=1):
        st.write(f"{i}. {fu}")

st.markdown("---")
st.info("For production: use managed secrets, index snapshotting, authentication and usage quotas.")
