"""
LexPal — AI Legal Information Assistant for India
Standalone Streamlit App · Powered by Groq (Free)
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

logger = logging.getLogger("lexpal")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexPal · Know Your Legal Rights",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Premium Design System ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ─ Reset & hide Streamlit UI chrome ─ */
#MainMenu, footer, header, .stDeployButton,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; visibility: hidden !important; }

/* ─ Base ─ */
:root {
    --bg: #030712;
    --surface: #0a0f1e;
    --surface-2: #111827;
    --border: rgba(255,255,255,0.05);
    --border-active: rgba(99,102,241,0.4);
    --text-primary: #f9fafb;
    --text-secondary: #9ca3af;
    --text-dim: #6b7280;
    --indigo: #6366f1;
    --violet: #8b5cf6;
    --cyan: #22d3ee;
    --amber: #fbbf24;
    --emerald: #34d399;
    --rose: #fb7185;
    --gradient: linear-gradient(135deg, #6366f1, #8b5cf6, #22d3ee);
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
    color: var(--text-primary) !important;
    background: var(--bg) !important;
}

.block-container {
    max-width: 860px !important;
    padding: 1rem 1.5rem 3rem !important;
}

/* ─ Hero ─ */
.hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 100px;
    padding: 6px 18px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #a5b4fc;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: -0.03em;
    margin-bottom: 1rem;
    background: linear-gradient(135deg, #f9fafb 0%, #a5b4fc 50%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1.08rem;
    color: var(--text-secondary);
    line-height: 1.7;
    max-width: 580px;
    margin: 0 auto 1.5rem;
    font-weight: 400;
}
.hero-features {
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
    margin-top: 1.5rem;
}
.hero-feat {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.82rem;
    color: var(--text-dim);
    font-weight: 500;
}
.hero-feat span.icon {
    font-size: 1rem;
}

/* ─ Disclaimer ─ */
.disc {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: rgba(251,191,36,0.05);
    border: 1px solid rgba(251,191,36,0.12);
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 0.78rem;
    color: #fcd34d;
    line-height: 1.6;
    margin: 0.5rem 0 1.5rem;
}

/* ─ Stats bar ─ */
.stats {
    display: flex;
    gap: 1px;
    background: var(--border);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.stat {
    flex: 1;
    background: var(--surface);
    padding: 14px 12px;
    text-align: center;
}
.stat-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
}
.stat-lbl {
    font-size: 0.68rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 3px;
}

/* ─ Tabs ─ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    padding: 4px !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    color: var(--text-dim) !important;
    font-weight: 500 !important;
    font-size: 0.84rem !important;
    padding: 10px 20px !important;
    transition: all 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-secondary) !important;
    background: rgba(255,255,255,0.03) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--indigo) !important;
    color: #fff !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ─ Inputs ─ */
.stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    padding: 14px !important;
    transition: border-color 0.2s !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus {
    border-color: var(--border-active) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}
.stTextArea textarea::placeholder {
    color: var(--text-dim) !important;
}

/* ─ Select ─ */
.stSelectbox > div > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
}

/* ─ Buttons ─ */
.stButton > button {
    background: var(--indigo) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.35) !important;
    filter: brightness(1.1) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ─ File uploader ─ */
[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}

/* ─ Answer card ─ */
.answer-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--indigo);
    border-radius: 12px;
    padding: 1.3rem 1.5rem;
    margin: 1rem 0;
}
.answer-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--indigo);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 10px;
}
.answer-body {
    font-size: 0.9rem;
    line-height: 1.9;
    color: var(--text-primary);
    white-space: pre-wrap;
    word-wrap: break-word;
}

/* ─ Source tags ─ */
.src-tags { display: flex; flex-wrap: wrap; gap: 6px; margin: 0.8rem 0; }
.src-tag {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.18);
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 0.74rem;
    color: #a5b4fc;
    font-weight: 500;
}

/* ─ Info box ─ */
.info-box {
    background: rgba(34,211,238,0.04);
    border: 1px solid rgba(34,211,238,0.12);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #67e8f9;
    line-height: 1.6;
    margin-top: 0.8rem;
}
.info-box a { color: #67e8f9; }

/* ─ Code/preview block ─ */
.preview-block {
    background: rgba(0,0,0,0.3);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
    font-family: 'SF Mono', 'Cascadia Code', monospace;
    font-size: 0.76rem;
    color: var(--text-secondary);
    max-height: 200px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.65;
}

/* ─ Expander ─ */
details summary {
    font-size: 0.86rem !important;
    font-weight: 500 !important;
}

/* ─ Divider ─ */
hr { border-color: var(--border) !important; opacity: 0.4 !important; }

/* ─ Footer ─ */
.foot {
    text-align: center;
    padding: 2.5rem 0 0.5rem;
    font-size: 0.72rem;
    color: var(--text-dim);
    letter-spacing: 0.03em;
}
.foot a { color: var(--indigo); text-decoration: none; }
.foot a:hover { color: var(--cyan); }

/* ─ Setup card ─ */
.setup-box {
    background: var(--surface);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 14px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 2rem 0;
}
.setup-box h3 {
    font-family: 'Space Grotesk', sans-serif;
    margin-bottom: 0.8rem;
}
.setup-box p {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.8;
}
.setup-box code {
    background: rgba(0,0,0,0.4);
    padding: 2px 8px;
    border-radius: 5px;
    font-size: 0.82rem;
}

/* ─ About section ─ */
.about-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 8px;
}
.about-label {
    font-size: 0.68rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.about-val {
    font-weight: 600;
    font-size: 0.88rem;
    color: var(--text-primary);
    margin-top: 2px;
}

/* ─ How it works steps ─ */
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 12px 0;
    border-bottom: 1px solid var(--border);
}
.step-row:last-child { border-bottom: none; }
.step-num {
    flex-shrink: 0;
    width: 28px; height: 28px;
    background: var(--indigo);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    color: #fff;
}
.step-text {
    font-size: 0.86rem;
    color: var(--text-secondary);
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ─── Cached Resources ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource(show_spinner=False)
def get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"},
    )


# ─── Core Logic ──────────────────────────────────────────────────────────────

def embed_query(text: str) -> list[float]:
    return load_embed_model().encode([text]).tolist()[0]

def retrieve_chunks(query: str, n: int = TOP_K) -> list[dict]:
    coll = get_collection()
    if coll.count() == 0:
        return []
    results = coll.query(
        query_embeddings=[embed_query(query)],
        n_results=min(n, coll.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": d, "metadata": m, "distance": dist}
        for d, m, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0],
        )
    ]

def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        parts.append(
            f"[Source {i}] {m.get('act', '?')} | {m.get('section', '?')} | Page {m.get('page', '?')}\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)

def call_groq(system_prompt: str, user_message: str) -> str:
    key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    if not key:
        return "❌ GROQ_API_KEY not set. Add it in Streamlit Cloud Secrets."
    try:
        r = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
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
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error calling Groq: {e}"

def run_ingestion():
    from utils.extractor import extract_pdf
    from utils.chunker import chunk_document
    coll = get_collection()
    model = load_embed_model()
    pdfs = list(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        return 0, "No PDFs found in data/ folder."
    total = 0
    for pdf in pdfs:
        try:
            text = extract_pdf(pdf)
            act_name = pdf.stem.replace("_", " ").replace("-", " ").title()
            chunks = chunk_document(text, act_name, CHUNK_SIZE, CHUNK_OVERLAP)
            if not chunks:
                continue
            texts = [c.text for c in chunks]
            metas = [c.to_metadata() for c in chunks]
            ids = [c.chunk_id for c in chunks]
            embs = model.encode(texts, show_progress_bar=False).tolist()
            for i in range(0, len(texts), 100):
                coll.upsert(
                    ids=ids[i:i+100], documents=texts[i:i+100],
                    embeddings=embs[i:i+100], metadatas=metas[i:i+100],
                )
            total += len(chunks)
        except Exception as e:
            logger.error(f"Failed to ingest {pdf.name}: {e}")
    return total, f"Ingested {total} chunks from {len(pdfs)} PDF(s)."


# ═══════════════════════════════════════════════════════════════════════════════
# HERO SECTION
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
    <div class="hero-badge">Free · Open Source · India</div>
    <div class="hero-title">Know Your Legal Rights</div>
    <div class="hero-sub">
        Ask any legal question or upload a document — get a clear,
        plain-language explanation with the exact laws and sections that apply.
        No jargon. No cost. Every answer cites its source.
    </div>
    <div class="hero-features">
        <div class="hero-feat"><span class="icon">📜</span> 5 Indian Acts</div>
        <div class="hero-feat"><span class="icon">🌐</span> English · Hindi · Marathi</div>
        <div class="hero-feat"><span class="icon">📎</span> PDF / Photo Upload</div>
        <div class="hero-feat"><span class="icon">✅</span> Source Citations</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DISCLAIMER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="disc">
    <span style="font-size:1.1rem;">⚠️</span>
    <span><strong>Legal information only.</strong> LexPal is not a lawyer. For criminal charges,
    court deadlines, or claims above ₹20 lakh, please consult a licensed advocate.</span>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# API KEY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
if not _key:
    st.markdown("""
    <div class="setup-box">
        <h3>🔑 One-Time Setup</h3>
        <p>
            LexPal uses Groq's free API (no credit card needed).<br><br>
            <strong>Step 1:</strong> Go to <a href="https://console.groq.com" target="_blank" style="color:#a5b4fc;">console.groq.com</a> → sign up free<br>
            <strong>Step 2:</strong> Create an API key<br>
            <strong>Step 3:</strong> Add <code>GROQ_API_KEY = "your_key"</code> to Streamlit Secrets
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# STATUS BAR
# ═══════════════════════════════════════════════════════════════════════════════

try:
    _coll = get_collection()
    _db_count = _coll.count()
except Exception:
    _db_count = 0

st.markdown(f"""
<div class="stats">
    <div class="stat"><div class="stat-val">🟢</div><div class="stat-lbl">AI Online</div></div>
    <div class="stat"><div class="stat-val">{_db_count:,}</div><div class="stat-lbl">Law Chunks</div></div>
    <div class="stat"><div class="stat-val">{len(COVERED_ACTS)}</div><div class="stat-lbl">Acts</div></div>
    <div class="stat"><div class="stat-val">3</div><div class="stat-lbl">Languages</div></div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask", "📄 Analyze", "⚙️ Database", "ℹ️ About"])

# ─── TAB 1: ASK ──────────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### Ask a Legal Question")
    st.caption("Type your question below. LexPal will find the relevant laws and explain them in simple language.")

    question = st.text_area(
        "question",
        placeholder="Example: A cheque given to me bounced. What can I do legally?\nExample: What are my rights if my landlord refuses to return the deposit?\nExample: How do I file an RTI application?",
        height=130,
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns([3, 1])
    with col_a:
        lang = st.selectbox("Reply in", SUPPORTED_LANGUAGES, key="ask_lang")
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        ask_btn = st.button("⚖️ Get Answer", use_container_width=True, key="ask_btn")

    if ask_btn:
        if not question.strip():
            st.warning("Please type a question first.")
        elif _db_count == 0:
            st.warning("📚 The law database is empty. Go to the **Database** tab and ingest some law PDFs first.")
        else:
            with st.spinner("Searching through Indian laws…"):
                chunks = retrieve_chunks(question, TOP_K)
                context = build_context(chunks)
                system = RAG_SYSTEM_PROMPT.format(language=lang, context=context)
                answer = call_groq(system, question)

            # ── Answer
            st.markdown(f"""
            <div class="answer-card">
                <div class="answer-label">LexPal's Answer</div>
                <div class="answer-body">{answer}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Sources
            if chunks:
                tags = "".join(
                    f'<span class="src-tag">📜 {c["metadata"].get("act","?")} · {c["metadata"].get("section","?")} · p.{c["metadata"].get("page","?")}</span>'
                    for c in chunks
                )
                st.markdown(f'<div style="margin-top:4px;"><strong style="font-size:0.78rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.08em;">Sources Used</strong></div><div class="src-tags">{tags}</div>', unsafe_allow_html=True)

                with st.expander("📖 View source excerpts"):
                    for i, c in enumerate(chunks, 1):
                        m = c["metadata"]
                        st.caption(f"**Source {i}** · {m.get('act','')} · {m.get('section','')} · Page {m.get('page','')}")
                        st.markdown(f'<div class="preview-block">{c["text"][:400]}</div>', unsafe_allow_html=True)
                        st.markdown("")

            # ── Lawyer note
            st.markdown("""
            <div class="info-box">
                👨‍⚖️ <strong>Need a lawyer?</strong> For court deadlines, criminal charges, or large claims —
                find one via the <a href="https://www.barcouncilofindia.org" target="_blank">Bar Council of India</a>.
            </div>
            """, unsafe_allow_html=True)


# ─── TAB 2: ANALYZE ──────────────────────────────────────────────────────────
with tab2:
    st.markdown("#### Analyze a Legal Document")
    st.caption("Upload a notice, agreement, court order, FIR copy, or a photo of one. LexPal will extract the text, find relevant laws, and tell you what to do next.")

    file = st.file_uploader(
        "upload",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png", "bmp", "tiff"],
        label_visibility="collapsed",
    )

    col_c, col_d = st.columns([3, 1])
    with col_c:
        lang2 = st.selectbox("Reply in", SUPPORTED_LANGUAGES, key="doc_lang")
    with col_d:
        st.markdown("<br>", unsafe_allow_html=True)
        doc_btn = st.button("🔎 Analyze", use_container_width=True, key="doc_btn")

    if doc_btn:
        if not file:
            st.warning("Upload a file first.")
        else:
            suffix = Path(file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file.getvalue())
                tmp_path = Path(tmp.name)

            try:
                with st.spinner("Extracting text…"):
                    from utils.extractor import extract_text
                    extracted = extract_text(tmp_path)

                if not extracted.strip():
                    st.error("Could not extract readable text from this file. Try a clearer scan or a different format.")
                else:
                    with st.expander("📄 Extracted text preview"):
                        st.markdown(f'<div class="preview-block">{extracted[:3000]}</div>', unsafe_allow_html=True)

                    with st.spinner("Analyzing against Indian laws…"):
                        chunks = retrieve_chunks(extracted[:1000], TOP_K)
                        context = build_context(chunks) if chunks else "No relevant laws found in database."
                        system = ANALYZE_SYSTEM_PROMPT.format(
                            language=lang2,
                            document_text=extracted[:3000],
                            context=context,
                        )
                        analysis = call_groq(system, "Analyze this document and give practical next steps.")

                    st.markdown(f"""
                    <div class="answer-card">
                        <div class="answer-label">Document Analysis</div>
                        <div class="answer-body">{analysis}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if chunks:
                        tags = "".join(
                            f'<span class="src-tag">📜 {c["metadata"].get("act","?")} · {c["metadata"].get("section","?")} · p.{c["metadata"].get("page","?")}</span>'
                            for c in chunks
                        )
                        st.markdown(f'<div style="margin-top:4px;"><strong style="font-size:0.78rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.08em;">Relevant Laws</strong></div><div class="src-tags">{tags}</div>', unsafe_allow_html=True)

                    st.markdown("""
                    <div class="info-box">
                        👨‍⚖️ <strong>Before you respond</strong> to any legal notice or court order,
                        consult a licensed lawyer. This analysis is a starting point, not a legal opinion.
                    </div>
                    """, unsafe_allow_html=True)
            finally:
                tmp_path.unlink(missing_ok=True)


# ─── TAB 3: DATABASE ─────────────────────────────────────────────────────────
with tab3:
    st.markdown("#### Law Database")
    st.caption("Place official bare-act PDFs in the `data/` folder, then run ingestion to process them into the vector database.")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Chunks in DB", f"{_db_count:,}")
    with c2:
        pdfs = list(DATA_DIR.glob("*.pdf")) if DATA_DIR.exists() else []
        st.metric("PDFs in data/", len(pdfs))

    if pdfs:
        st.markdown("**Files found:**")
        for p in pdfs:
            st.text(f"  📄 {p.name}  ({p.stat().st_size / 1024:.0f} KB)")

    st.divider()
    st.markdown("**Recommended acts to download (official sources):**")
    st.markdown("""
| Act | Source |
|-----|--------|
| Consumer Protection Act 2019 | [consumeraffairs.nic.in](https://consumeraffairs.nic.in) |
| Negotiable Instruments Act 1881 | [indiacode.nic.in](https://indiacode.nic.in) |
| IT Act 2000 | [indiacode.nic.in](https://indiacode.nic.in) |
| RTI Act 2005 | [rti.gov.in](https://rti.gov.in) |
| Maharashtra Rent Control Act 1999 | [maharerait.mahaonline.gov.in](https://maharerait.mahaonline.gov.in) |
    """)

    st.divider()
    if st.button("🚀 Run Ingestion", use_container_width=True, key="ingest_btn"):
        with st.spinner("Processing PDFs… this may take a few minutes."):
            total, msg = run_ingestion()
        if total > 0:
            st.success(f"✅ {msg}")
            st.rerun()
        else:
            st.warning(msg)


# ─── TAB 4: ABOUT ────────────────────────────────────────────────────────────
with tab4:
    st.markdown("#### About LexPal")

    st.markdown("""
> **LexPal** is a free, open-source AI assistant that helps ordinary people
> understand their legal rights in India — in plain English, Hindi, or Marathi.
> Every answer is sourced from official law texts, not AI guesswork.
    """)

    st.markdown("**How it works:**")
    st.markdown("""
<div style="background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:16px 20px;">
    <div class="step-row"><div class="step-num">1</div><div class="step-text">Official law PDFs are chunked by section and tagged with Act name, Section number & page</div></div>
    <div class="step-row"><div class="step-num">2</div><div class="step-text">Each chunk is converted to a vector and stored in a local ChromaDB database</div></div>
    <div class="step-row"><div class="step-num">3</div><div class="step-text">Your question is matched against the database to find the most relevant law chunks</div></div>
    <div class="step-row"><div class="step-num">4</div><div class="step-text">An AI model (Llama 3 via Groq) generates an answer using <strong>only</strong> those chunks</div></div>
    <div class="step-row"><div class="step-num">5</div><div class="step-text">If the answer isn't in the database, LexPal tells you honestly instead of guessing</div></div>
</div>
    """, unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Laws Covered**")
        for act in COVERED_ACTS:
            st.markdown(f"- {act}")
    with c2:
        st.markdown("**Tech Stack**")
        st.markdown("- **LLM:** Groq · Llama 3 (free)")
        st.markdown("- **Embeddings:** all-MiniLM-L6-v2")
        st.markdown("- **Vector DB:** ChromaDB")
        st.markdown("- **OCR:** Tesseract")
        st.markdown("- **Frontend:** Streamlit")

    st.divider()
    st.markdown("""
🔒 **Privacy:** Uploaded documents are processed in memory only and never stored.
No queries or personal data are saved.

⚠️ **Disclaimer:** LexPal provides legal information, not legal advice.
It may be incorrect or outdated. Always consult a licensed lawyer for serious matters.
    """)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="foot">
    ⚖️ LexPal · Legal Information Only · 100% Free ·
    <a href="https://github.com/jishanmujawar33/LexPal" target="_blank">GitHub</a> ·
    <a href="https://www.barcouncilofindia.org" target="_blank">Find a Lawyer</a>
</div>
""", unsafe_allow_html=True)
