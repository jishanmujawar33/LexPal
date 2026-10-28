"""
LexPal – AI-Powered Legal Information Assistant
Standalone Streamlit App (no FastAPI dependency)
"""

import os
import tempfile
import logging
from pathlib import Path

import streamlit as st
import httpx

from config import (
    GROQ_API_KEY, GROQ_MODEL, CHROMA_DIR, CHROMA_COLLECTION,
    EMBED_MODEL, RAG_SYSTEM_PROMPT, ANALYZE_SYSTEM_PROMPT,
    DISCLAIMER, COVERED_ACTS, SUPPORTED_LANGUAGES, TOP_K, DATA_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lexpal")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexPal — AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Premium CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

/* ── Hide Streamlit chrome ─── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
[data-testid="stToolbar"] {display: none;}

/* ── Root ─── */
:root {
    --bg: #09090b;
    --card: rgba(255,255,255,0.03);
    --card-hover: rgba(255,255,255,0.06);
    --border: rgba(255,255,255,0.06);
    --border-hover: rgba(255,255,255,0.12);
    --text: #fafafa;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;
    --accent-1: #6366f1;
    --accent-2: #8b5cf6;
    --accent-3: #06b6d4;
    --amber: #f59e0b;
    --amber-bg: rgba(245,158,11,0.08);
    --amber-border: rgba(245,158,11,0.2);
    --emerald: #10b981;
    --rose: #f43f5e;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.block-container {
    max-width: 920px !important;
    padding: 2rem 1.5rem 4rem !important;
}

/* ── Animated gradient header ─── */
.lex-header {
    text-align: center;
    padding: 2.5rem 0 1rem;
    position: relative;
}
.lex-logo {
    font-family: 'Outfit', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 30%, #06b6d4 70%, #34d399 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-shift 6s ease infinite;
}
@keyframes gradient-shift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
.lex-tagline {
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 400;
    margin-top: 0.3rem;
    letter-spacing: 0.02em;
}

/* ── Glass cards ─── */
.glass-card {
    background: var(--card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
    background: var(--card-hover);
    border-color: var(--border-hover);
}
.glass-card-accent {
    border-left: 3px solid var(--accent-1);
}

/* ── Disclaimer banner ─── */
.disclaimer-bar {
    background: var(--amber-bg);
    border: 1px solid var(--amber-border);
    border-radius: 12px;
    padding: 0.7rem 1.2rem;
    margin-bottom: 1.5rem;
    font-size: 0.82rem;
    color: #fbbf24;
    display: flex;
    align-items: center;
    gap: 8px;
    line-height: 1.5;
}

/* ── Metric cards ─── */
.metric-row {
    display: flex;
    gap: 12px;
    margin: 1rem 0 1.5rem;
}
.metric-card {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: all 0.2s ease;
}
.metric-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-2px);
}
.metric-value {
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-3));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── Source pills ─── */
.source-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.76rem;
    color: #a5b4fc;
    margin: 3px 4px 3px 0;
    transition: all 0.2s ease;
    font-weight: 500;
}
.source-pill:hover {
    background: rgba(99,102,241,0.18);
    border-color: rgba(99,102,241,0.4);
}

/* ── Answer block ─── */
.answer-block {
    font-size: 0.92rem;
    line-height: 1.85;
    color: var(--text);
    white-space: pre-wrap;
}
.answer-block strong { color: #c4b5fd; }

/* ── Lawyer callout ─── */
.lawyer-box {
    background: rgba(16,185,129,0.06);
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 12px;
    padding: 0.8rem 1.1rem;
    font-size: 0.84rem;
    color: #6ee7b7;
    margin-top: 1rem;
}
.lawyer-box a { color: #6ee7b7; text-decoration: underline; }

/* ── Extracted text ─── */
.code-preview {
    background: rgba(0,0,0,0.4);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.78rem;
    color: var(--text-secondary);
    max-height: 220px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.6;
}

/* ── Section titles ─── */
.section-title {
    font-family: 'Outfit', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text);
    margin: 1.2rem 0 0.6rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.02) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-secondary) !important;
    background: rgba(255,255,255,0.03) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
    color: white !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Inputs ─── */
.stTextArea textarea, .stTextInput input {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    transition: all 0.2s ease !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.stTextArea textarea::placeholder { color: var(--text-muted) !important; }

/* ── Select ─── */
.stSelectbox [data-baseweb="select"] {
    background: rgba(255,255,255,0.03) !important;
    border-color: var(--border) !important;
    border-radius: 12px !important;
}
.stSelectbox [data-baseweb="select"]:hover {
    border-color: var(--border-hover) !important;
}

/* ── Buttons ─── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.01em !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
    filter: brightness(1.1) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── File uploader ─── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 2px dashed rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(99,102,241,0.3) !important;
}

/* ── Expander ─── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.02) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

/* ── Dividers ─── */
hr { border-color: var(--border) !important; opacity: 0.5 !important; }

/* ── Footer ─── */
.lex-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: var(--text-muted);
    font-size: 0.78rem;
    letter-spacing: 0.02em;
}
.lex-footer a {
    color: var(--accent-1);
    text-decoration: none;
    transition: color 0.2s ease;
}
.lex-footer a:hover { color: var(--accent-3); }

/* ── Spinner ─── */
.stSpinner > div { border-top-color: var(--accent-1) !important; }

/* ── About cards ─── */
.about-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin: 1rem 0;
}
.about-item {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
}
.about-item-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.about-item-value {
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text);
}

/* ── Setup guide ─── */
.setup-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(6,182,212,0.05));
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 2rem 0;
}
.setup-card h3 {
    font-family: 'Outfit', sans-serif;
    color: var(--text);
    margin-bottom: 0.5rem;
}
.setup-card p { color: var(--text-secondary); font-size: 0.9rem; }
.setup-card code {
    background: rgba(0,0,0,0.3);
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ── Cached Resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource(show_spinner=False)
def get_chroma_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

# ── Core Functions ────────────────────────────────────────────────────────────

def embed_query(text: str) -> list[float]:
    model = load_embedding_model()
    return model.encode([text]).tolist()[0]

def retrieve_chunks(query: str, n: int = TOP_K) -> list[dict]:
    collection = get_chroma_collection()
    if collection.count() == 0:
        return []
    q_emb = embed_query(query)
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=min(n, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({"text": doc, "metadata": meta, "distance": dist})
    return chunks

def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        parts.append(
            f"[Source {i}] {m.get('act', '?')} | {m.get('section', '?')} | Page {m.get('page', '?')}\n"
            f"{c['text']}"
        )
    return "\n\n---\n\n".join(parts)

def call_groq(system_prompt: str, user_message: str) -> str:
    api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    if not api_key:
        return "❌ GROQ_API_KEY is not configured. Please set it in Streamlit Cloud secrets or environment variables."
    try:
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.1,
                "max_tokens": 2048,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except httpx.ConnectError:
        return "❌ Cannot reach Groq API. Please check your internet connection."
    except Exception as e:
        return f"❌ Error: {e}"

def run_ingestion():
    from utils.extractor import extract_pdf
    from utils.chunker import chunk_document
    collection = get_chroma_collection()
    model = load_embedding_model()

    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        return 0, "No PDFs found in data/ folder."

    total = 0
    for pdf in pdfs:
        text = extract_pdf(pdf)
        act_name = pdf.stem.replace("_", " ").replace("-", " ").title()
        chunks = chunk_document(text, act_name, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            continue
        texts = [c.text for c in chunks]
        metas = [c.to_metadata() for c in chunks]
        ids = [c.chunk_id for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        batch = 100
        for i in range(0, len(texts), batch):
            collection.upsert(
                ids=ids[i:i+batch],
                documents=texts[i:i+batch],
                embeddings=embeddings[i:i+batch],
                metadatas=metas[i:i+batch],
            )
        total += len(chunks)
    return total, f"✅ Ingested {total} chunks from {len(pdfs)} PDF(s)."

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lex-header">
    <div class="lex-logo">⚖️ LexPal</div>
    <div class="lex-tagline">AI-Powered Legal Information Assistant · India</div>
</div>
""", unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer-bar">
    <span>⚠️</span>
    <span>LexPal provides legal <strong>information</strong>, not legal <strong>advice</strong>.
    For criminal charges, court deadlines, or large financial claims, consult a licensed lawyer.</span>
</div>
""", unsafe_allow_html=True)

# ── Check API key ─────────────────────────────────────────────────────────────
api_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
if not api_key:
    st.markdown("""
    <div class="setup-card">
        <h3>🔑 Setup Required</h3>
        <p>LexPal needs a free Groq API key to work.</p>
        <p style="margin-top:1rem;">
            1. Go to <a href="https://console.groq.com" target="_blank" style="color:#818cf8;">console.groq.com</a> (free signup, no credit card)<br>
            2. Create an API key<br>
            3. Add <code>GROQ_API_KEY</code> to your Streamlit Cloud secrets or environment
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Status metrics ────────────────────────────────────────────────────────────
try:
    collection = get_chroma_collection()
    db_count = collection.count()
except Exception:
    db_count = 0

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-value">{"●" if api_key else "○"}</div>
        <div class="metric-label">Groq API</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{db_count:,}</div>
        <div class="metric-label">Law Chunks</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{len(COVERED_ACTS)}</div>
        <div class="metric-label">Acts Covered</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">3</div>
        <div class="metric-label">Languages</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_ask, tab_analyze, tab_db, tab_about = st.tabs([
    "💬  Ask a Question",
    "📄  Analyze Document",
    "⚙️  Database",
    "ℹ️  About",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — ASK A QUESTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_ask:
    st.markdown('<div class="section-title">💬 Ask a Legal Question</div>', unsafe_allow_html=True)

    with st.form("ask_form", clear_on_submit=False):
        question = st.text_area(
            "Your question",
            placeholder="e.g. What are my rights if my landlord refuses to return my security deposit?",
            height=120,
            label_visibility="collapsed",
        )
        c1, c2 = st.columns([3, 1])
        with c1:
            language = st.selectbox("Language", SUPPORTED_LANGUAGES, label_visibility="collapsed")
        with c2:
            asked = st.form_submit_button("Ask LexPal", use_container_width=True)

    if asked and question.strip():
        with st.spinner("Searching laws & generating answer…"):
            chunks = retrieve_chunks(question, TOP_K)
            if not chunks:
                st.warning("📚 No law data in the database yet. Go to the **Database** tab to ingest law PDFs first.")
            else:
                context = build_context(chunks)
                system = RAG_SYSTEM_PROMPT.format(language=language, context=context)
                answer = call_groq(system, question)

                # Answer card
                st.markdown(f"""
                <div class="glass-card glass-card-accent">
                    <div class="section-title">📋 Answer</div>
                    <div class="answer-block">{answer}</div>
                </div>
                """, unsafe_allow_html=True)

                # Source pills
                st.markdown('<div class="section-title">📌 Sources</div>', unsafe_allow_html=True)
                pills = ""
                for s in chunks:
                    m = s["metadata"]
                    pills += f'<span class="source-pill">📜 {m.get("act","?")} · {m.get("section","?")} · p.{m.get("page","?")}</span>'
                st.markdown(pills, unsafe_allow_html=True)

                with st.expander("View source excerpts"):
                    for i, s in enumerate(chunks, 1):
                        m = s["metadata"]
                        st.markdown(f"**Source {i}** — {m.get('act','')} | {m.get('section','')} | Page {m.get('page','')}")
                        st.markdown(f'<div class="code-preview">{s["text"][:500]}</div>', unsafe_allow_html=True)

                # Lawyer callout
                st.markdown("""
                <div class="lawyer-box">
                    👨‍⚖️ <strong>Need professional help?</strong> For court deadlines, criminal matters, or claims above ₹20 lakh,
                    consult a licensed lawyer via the <a href="https://www.barcouncilofindia.org" target="_blank">Bar Council of India</a>.
                </div>
                """, unsafe_allow_html=True)
    elif asked:
        st.warning("Please enter a question.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — ANALYZE DOCUMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_analyze:
    st.markdown('<div class="section-title">📄 Upload & Analyze a Legal Document</div>', unsafe_allow_html=True)
    st.caption("Upload a notice, agreement, court order, FIR copy, or photo. Supported: PDF, DOCX, TXT, JPG, PNG (max 10 MB).")

    uploaded = st.file_uploader(
        "Upload document",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png", "bmp", "tiff"],
        label_visibility="collapsed",
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        lang2 = st.selectbox("Language", SUPPORTED_LANGUAGES, key="analyze_lang", label_visibility="collapsed")
    with c2:
        analyze_clicked = st.button("🔎 Analyze", use_container_width=True)

    if analyze_clicked:
        if not uploaded:
            st.warning("Please upload a file first.")
        else:
            with st.spinner("Extracting text & analyzing…"):
                suffix = Path(uploaded.name).suffix.lower()
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = Path(tmp.name)

                try:
                    from utils.extractor import extract_text
                    extracted = extract_text(tmp_path)

                    if not extracted.strip():
                        st.error("Could not extract text from this file.")
                    else:
                        with st.expander("📄 Extracted Text"):
                            st.markdown(f'<div class="code-preview">{extracted[:3000]}</div>', unsafe_allow_html=True)

                        chunks = retrieve_chunks(extracted[:1000], TOP_K)
                        context = build_context(chunks) if chunks else "No relevant laws found in database."
                        system = ANALYZE_SYSTEM_PROMPT.format(
                            language=lang2,
                            document_text=extracted[:3000],
                            context=context,
                        )
                        analysis = call_groq(system, "Analyze this document and provide next steps.")

                        st.markdown(f"""
                        <div class="glass-card glass-card-accent">
                            <div class="section-title">🧠 Analysis</div>
                            <div class="answer-block">{analysis}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        if chunks:
                            st.markdown('<div class="section-title">📌 Relevant Laws</div>', unsafe_allow_html=True)
                            pills = ""
                            for s in chunks:
                                m = s["metadata"]
                                pills += f'<span class="source-pill">📜 {m.get("act","?")} · {m.get("section","?")} · p.{m.get("page","?")}</span>'
                            st.markdown(pills, unsafe_allow_html=True)

                        st.markdown("""
                        <div class="lawyer-box">
                            👨‍⚖️ <strong>Important:</strong> This analysis is a starting point only.
                            Before responding to any legal notice or court order, consult a licensed lawyer.
                        </div>
                        """, unsafe_allow_html=True)
                finally:
                    tmp_path.unlink(missing_ok=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — DATABASE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_db:
    st.markdown('<div class="section-title">⚙️ Law Database</div>', unsafe_allow_html=True)
    st.caption("Place official bare-act PDFs in the `data/` folder, then click Ingest to process them into the vector database.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-value">{db_count}</div>
            <div class="metric-label" style="margin-top:4px;">Chunks in database</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        pdfs = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
        st.markdown(f"""
        <div class="glass-card">
            <div class="metric-value">{len(pdfs)}</div>
            <div class="metric-label" style="margin-top:4px;">PDFs in data/ folder</div>
        </div>
        """, unsafe_allow_html=True)

    if pdfs:
        st.markdown('<div class="section-title">📂 Files Found</div>', unsafe_allow_html=True)
        for p in pdfs:
            st.markdown(f"- `{p.name}` ({p.stat().st_size / 1024:.0f} KB)")

    st.markdown("---")
    st.markdown("**Recommended acts to download:**")
    for act in COVERED_ACTS:
        st.markdown(f"- {act}")

    st.markdown("---")
    if st.button("🚀 Run Ingestion", use_container_width=True):
        with st.spinner("Ingesting PDFs… this may take a few minutes."):
            total, msg = run_ingestion()
            if total > 0:
                st.success(msg)
                st.rerun()
            else:
                st.warning(msg)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — ABOUT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_about:
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:2rem;">
        <div style="font-family:'Outfit',sans-serif; font-size:1.8rem; font-weight:700;
             background:linear-gradient(135deg,#818cf8,#06b6d4); -webkit-background-clip:text;
             -webkit-text-fill-color:transparent; margin-bottom:0.5rem;">
            ⚖️ LexPal
        </div>
        <div style="color:var(--text-secondary); font-size:0.92rem; max-width:500px; margin:0 auto; line-height:1.7;">
            A free, open-source AI assistant that helps ordinary people understand
            their legal rights in India — in plain English, Hindi, or Marathi.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🧠 How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <ol style="color:var(--text-secondary); line-height:2; font-size:0.88rem; padding-left:1.2rem;">
            <li>Official bare-act PDFs are split into chunks tagged with Act, Section & page number</li>
            <li>Each chunk is converted to a vector and stored in ChromaDB (local database)</li>
            <li>When you ask a question, the most relevant law chunks are retrieved</li>
            <li>Groq's Llama 3 generates an answer using <strong>only</strong> those retrieved chunks</li>
            <li>Every answer cites its sources. If the answer isn't in the database, LexPal says so</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">📚 Laws Covered</div>', unsafe_allow_html=True)
    st.markdown('<div class="about-grid">', unsafe_allow_html=True)
    for act in COVERED_ACTS:
        parts = act.rsplit(" ", 1)
        name = parts[0] if len(parts) > 1 else act
        year = parts[1] if len(parts) > 1 else ""
        st.markdown(f"""
        <div class="about-item">
            <div class="about-item-label">{year}</div>
            <div class="about-item-value">{name}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔧 Tech Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="about-grid">
        <div class="about-item"><div class="about-item-label">LLM</div><div class="about-item-value">Groq · Llama 3</div></div>
        <div class="about-item"><div class="about-item-label">Embeddings</div><div class="about-item-value">all-MiniLM-L6-v2</div></div>
        <div class="about-item"><div class="about-item-label">Vector DB</div><div class="about-item-value">ChromaDB</div></div>
        <div class="about-item"><div class="about-item-label">OCR</div><div class="about-item-value">Tesseract</div></div>
        <div class="about-item"><div class="about-item-label">Frontend</div><div class="about-item-value">Streamlit</div></div>
        <div class="about-item"><div class="about-item-label">Cost</div><div class="about-item-value">100% Free</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="margin-top:1rem;">
        <div class="section-title">🔒 Privacy</div>
        <p style="color:var(--text-secondary); font-size:0.88rem; line-height:1.7; margin:0;">
            Uploaded documents are processed in memory only and deleted immediately.
            No user data, queries, or documents are stored permanently.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lex-footer">
    ⚖️ LexPal · Legal Information Only · Not Legal Advice · 100% Free & Open Source ·
    <a href="https://github.com/jishanmujawar33/LexPal" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)
