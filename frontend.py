"""
frontend.py  –  LexPal Streamlit Interface
===========================================
Run with:
    streamlit run frontend.py

Tabs:
  1. Ask a Question  – RAG-powered Q&A with citations
  2. Analyze My Document  – Upload PDF/DOCX/image → structured analysis
"""

import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexPal – AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_BASE = "http://localhost:8000"

DISCLAIMER_TEXT = (
    "⚠️ **DISCLAIMER:** LexPal provides legal **information**, not legal **advice**. "
    "For criminal charges, court deadlines, or large financial claims, "
    "please consult a licensed lawyer."
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');

/* ── Root variables ─── */
:root {
    --primary:   #4f46e5;
    --primary-light: #818cf8;
    --accent:    #06b6d4;
    --bg:        #0f0f1a;
    --card:      #1a1a2e;
    --card2:     #16213e;
    --border:    #2d2d52;
    --text:      #e2e8f0;
    --text-muted:#94a3b8;
    --danger:    #ef4444;
    --success:   #22c55e;
    --warn:      #f59e0b;
}

/* ── Global ─── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
.block-container { max-width: 1100px; padding: 1.5rem 2rem; }

/* ── Header ─── */
.lex-header {
    display: flex; align-items: center; gap: 14px;
    padding: 1.2rem 0 0.4rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1rem;
}
.lex-logo {
    font-family: 'Outfit', sans-serif;
    font-size: 2.1rem; font-weight: 700;
    background: linear-gradient(135deg, #818cf8 0%, #06b6d4 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.lex-tagline { color: var(--text-muted); font-size: 0.9rem; }

/* ── Disclaimer banner ─── */
.disclaimer-banner {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.4);
    border-left: 4px solid var(--danger);
    border-radius: 8px;
    padding: 0.75rem 1.1rem;
    margin-bottom: 1.2rem;
    font-size: 0.88rem;
    color: #fca5a5;
}

/* ── Cards ─── */
.lex-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.lex-card-accent {
    border-left: 4px solid var(--primary-light);
}

/* ── Answer text ─── */
.answer-text {
    font-size: 0.97rem;
    line-height: 1.75;
    white-space: pre-wrap;
}

/* ── Source chip ─── */
.source-chip {
    display: inline-block;
    background: rgba(79,70,229,0.2);
    border: 1px solid rgba(79,70,229,0.5);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: var(--primary-light);
    margin: 3px 4px 3px 0;
    cursor: default;
}

/* ── Lawyer callout ─── */
.lawyer-callout {
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.35);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    color: #fde68a;
    font-size: 0.88rem;
    margin-top: 1rem;
}

/* ── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card2) !important;
    border-radius: 10px !important;
    gap: 4px; padding: 4px;
    border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary) 0%, #7c3aed 100%) !important;
    color: white !important;
}

/* ── Inputs ─── */
.stTextArea textarea, .stTextInput input, .stSelectbox select {
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(79,70,229,0.2) !important;
}

/* ── Buttons ─── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.8rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(79,70,229,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79,70,229,0.45) !important;
}

/* ── File uploader ─── */
[data-testid="stFileUploader"] {
    background: var(--card2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    padding: 0.8rem !important;
}

/* ── Spinner ─── */
.stSpinner { color: var(--primary-light) !important; }

/* ── Extracted text box ─── */
.extracted-box {
    background: #0a0a14;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    font-family: 'Courier New', monospace;
    color: var(--text-muted);
    max-height: 200px;
    overflow-y: auto;
    white-space: pre-wrap;
}

/* ── Status badge ─── */
.badge-ok  { color: var(--success); font-weight:600; }
.badge-err { color: var(--danger);  font-weight:600; }

/* ── Section heading ─── */
.section-heading {
    font-family: 'Outfit', sans-serif;
    font-size: 1.05rem; font-weight: 600;
    color: var(--primary-light);
    margin: 1rem 0 0.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lex-header">
  <span class="lex-logo">⚖️ LexPal</span>
  <div>
    <div style="font-weight:600;font-size:1rem;">AI Legal Information Assistant</div>
    <div class="lex-tagline">India · Consumer, Tenancy, IT, NI Act, RTI · Free & Open Source</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Sticky disclaimer ─────────────────────────────────────────────────────────
st.markdown(f'<div class="disclaimer-banner">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)


# ── API health check ──────────────────────────────────────────────────────────
def check_api() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=4)
        return r.json() if r.ok else None
    except Exception:
        return None


health = check_api()
if health:
    db_chunks = health.get("db_chunks", 0)
    model     = health.get("model", "?")
    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Backend", "Online")
    col2.metric("📚 Law Chunks", f"{db_chunks:,}")
    col3.metric("🤖 Model", model.split(":")[0])
else:
    st.error(
        "⚠️ Backend is offline. Start it with: `uvicorn main:app --reload`",
        icon="🔴",
    )

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_ask, tab_analyze, tab_ingest, tab_about = st.tabs([
    "💬 Ask a Question",
    "📄 Analyze My Document",
    "⚙️ Ingest Laws",
    "ℹ️ About",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – ASK A QUESTION
# ─────────────────────────────────────────────────────────────────────────────
with tab_ask:
    st.markdown('<div class="section-heading">Ask a Legal Question</div>', unsafe_allow_html=True)
    st.caption("Questions are answered using only the laws in the database. Every answer includes citations.")

    with st.form("ask_form"):
        question = st.text_area(
            "Your Question",
            placeholder=(
                "e.g. What are my rights if my landlord refuses to return the security deposit?\n"
                "e.g. A cheque given to me bounced. What can I do?\n"
                "e.g. How do I file an RTI application?"
            ),
            height=130,
            key="question_input",
        )
        col_lang, col_btn = st.columns([2, 1])
        with col_lang:
            language = st.selectbox("Reply Language", ["English", "Hindi", "Marathi"], key="ask_lang")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🔍 Get Answer", use_container_width=True)

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching laws and generating answer…"):
                try:
                    resp = requests.post(
                        f"{API_BASE}/ask",
                        json={"question": question, "language": language},
                        timeout=120,
                    )
                    if resp.ok:
                        data = resp.json()
                        # ── Answer card ──────────────────────────────────────
                        st.markdown('<div class="lex-card lex-card-accent">', unsafe_allow_html=True)
                        st.markdown("**📋 Answer**")
                        st.markdown(
                            f'<div class="answer-text">{data["answer"]}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                        # ── Sources ──────────────────────────────────────────
                        sources = data.get("sources", [])
                        if sources:
                            st.markdown('<div class="section-heading">📌 Sources Used</div>', unsafe_allow_html=True)
                            chips = ""
                            for s in sources:
                                chips += f'<span class="source-chip">📜 {s["act"]} · {s["section"]} · p.{s["page"]}</span>'
                            st.markdown(chips, unsafe_allow_html=True)

                            with st.expander("View source snippets"):
                                for i, s in enumerate(sources, 1):
                                    st.markdown(f"**Source {i}** – {s['act']} | {s['section']} | Page {s['page']}")
                                    st.markdown(
                                        f'<div class="extracted-box">{s["snippet"]}</div>',
                                        unsafe_allow_html=True,
                                    )

                        # ── Lawyer callout ───────────────────────────────────
                        st.markdown(
                            '<div class="lawyer-callout">'
                            '👨‍⚖️ <strong>Need professional help?</strong> '
                            'For court deadlines, criminal matters, or claims above ₹20 lakh, '
                            'please consult a licensed lawyer. '
                            'You can find one via the <a href="https://www.barcouncilofindia.org" target="_blank" '
                            'style="color:#fde68a">Bar Council of India</a>.'
                            '</div>',
                            unsafe_allow_html=True,
                        )

                        # ── Disclaimer ───────────────────────────────────────
                        st.info(data.get("disclaimer", ""), icon="⚠️")

                    else:
                        st.error(f"API Error {resp.status_code}: {resp.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the backend. Make sure `uvicorn main:app --reload` is running.")
                except Exception as e:
                    st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – ANALYZE DOCUMENT
# ─────────────────────────────────────────────────────────────────────────────
with tab_analyze:
    st.markdown('<div class="section-heading">Upload & Analyze a Legal Document</div>', unsafe_allow_html=True)
    st.caption(
        "Upload a rent notice, court order, FIR copy, cheque bounce notice, agreement, or any legal document. "
        "Supported: PDF, DOCX, TXT, JPG, PNG (max 10 MB)."
    )

    uploaded_file = st.file_uploader(
        "Drop your document here",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png", "bmp", "tiff"],
        key="doc_uploader",
        label_visibility="collapsed",
    )

    col_lang2, col_analyze_btn = st.columns([2, 1])
    with col_lang2:
        lang2 = st.selectbox("Reply Language", ["English", "Hindi", "Marathi"], key="analyze_lang")
    with col_analyze_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_clicked = st.button("🔎 Analyze Document", use_container_width=True, key="analyze_btn")

    if analyze_clicked:
        if not uploaded_file:
            st.warning("Please upload a file first.")
        else:
            with st.spinner("Extracting text and analyzing document…"):
                try:
                    files   = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    payload = {"language": lang2}
                    resp = requests.post(
                        f"{API_BASE}/analyze",
                        files=files,
                        data=payload,
                        timeout=180,
                    )
                    if resp.ok:
                        data = resp.json()

                        # ── OCR badge ──────────────────────────────────────
                        if data.get("ocr_used"):
                            st.success("✅ OCR was used to extract text from this document.", icon="🔍")

                        # ── Extracted text preview ─────────────────────────
                        with st.expander("📄 Extracted Document Text"):
                            st.markdown(
                                f'<div class="extracted-box">{data["extracted_text"][:3000]}</div>',
                                unsafe_allow_html=True,
                            )

                        # ── Analysis card ──────────────────────────────────
                        st.markdown('<div class="lex-card lex-card-accent">', unsafe_allow_html=True)
                        st.markdown("**🧠 Document Analysis**")
                        st.markdown(
                            f'<div class="answer-text">{data["analysis"]}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                        # ── Sources ────────────────────────────────────────
                        sources = data.get("sources", [])
                        if sources:
                            st.markdown('<div class="section-heading">📌 Relevant Laws Found</div>', unsafe_allow_html=True)
                            chips = ""
                            for s in sources:
                                chips += f'<span class="source-chip">📜 {s["act"]} · {s["section"]} · p.{s["page"]}</span>'
                            st.markdown(chips, unsafe_allow_html=True)

                        # ── Lawyer callout ─────────────────────────────────
                        st.markdown(
                            '<div class="lawyer-callout">'
                            '👨‍⚖️ <strong>Important:</strong> This analysis is a starting point only. '
                            'Before responding to any legal notice or court order, '
                            'consult a licensed lawyer.'
                            '</div>',
                            unsafe_allow_html=True,
                        )
                        st.info(data.get("disclaimer", ""), icon="⚠️")

                    else:
                        detail = resp.json().get("detail", resp.text)
                        st.error(f"Error {resp.status_code}: {detail}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the backend. Make sure `uvicorn main:app --reload` is running.")
                except Exception as e:
                    st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – INGEST LAWS
# ─────────────────────────────────────────────────────────────────────────────
with tab_ingest:
    st.markdown('<div class="section-heading">⚙️ Ingest Law PDFs</div>', unsafe_allow_html=True)
    st.info(
        "Place official bare act PDFs in the `data/` folder, then click **Run Ingestion**. "
        "This processes every PDF, chunks it section-by-section, embeds it, and stores it in ChromaDB. "
        "Run this once after downloading laws.",
        icon="💡",
    )

    st.markdown("**Recommended PDFs to download:**")
    acts = [
        ("Consumer Protection Act 2019", "https://consumeraffairs.nic.in/sites/default/files/file-uploads/acts-and-rules/Consumer%20Protection%20Act%202019.pdf"),
        ("Negotiable Instruments Act 1881", "https://indiacode.nic.in/bitstream/123456789/2051/1/188126.pdf"),
        ("IT Act 2000", "https://indiacode.nic.in/bitstream/123456789/1999/3/200021.pdf"),
        ("RTI Act 2005", "https://rti.gov.in/rti-act.pdf"),
        ("Maharashtra Rent Control Act 1999", "https://maharerait.mahaonline.gov.in/images/pdf/Maharashtra_Rent_Control_act_1999.pdf"),
    ]
    for name, url in acts:
        st.markdown(f"- [{name}]({url})")

    st.markdown("---")
    if st.button("🚀 Run Ingestion Pipeline", key="ingest_btn"):
        with st.spinner("Ingesting PDFs into ChromaDB… this may take a few minutes."):
            try:
                resp = requests.post(f"{API_BASE}/ingest", timeout=600)
                if resp.ok:
                    st.success(f"✅ {resp.json().get('message', 'Done!')}")
                    st.rerun()
                else:
                    st.error(f"Ingestion failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown('<div class="section-heading">About LexPal</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="lex-card">

**LexPal** is a free, open-source AI assistant that helps ordinary people understand their legal rights in India — in plain English, Hindi, or Marathi.

### How It Works (RAG)
1. Official bare-act PDFs are split into chunks tagged with Act name, Section number, and page.
2. Each chunk is converted to a vector and stored in ChromaDB (local, no cloud).
3. When you ask a question or upload a document, the most relevant law chunks are retrieved.
4. A local LLM (Llama 3.1 via Ollama) generates an answer using **only** those chunks.
5. Every answer cites its sources. If the answer is not in the database, LexPal says so.

### Laws Covered (MVP)
| Act | Year |
|-----|------|
| Consumer Protection Act | 2019 |
| Negotiable Instruments Act | 1881 |
| Information Technology Act | 2000 |
| Maharashtra Rent Control Act | 1999 |
| Right to Information Act | 2005 |

### Privacy
- Uploaded documents are processed **in memory only** and deleted immediately.
- No user data, queries, or documents are stored.

### Tech Stack
| Layer | Tool |
|-------|------|
| LLM | Ollama + Llama 3.1 8B |
| Embeddings | all-MiniLM-L6-v2 |
| Vector DB | ChromaDB (local) |
| OCR | Tesseract (eng+hin+mar) |
| Backend | FastAPI |
| Frontend | Streamlit |

### Limitations & Disclaimer
LexPal provides **legal information, not legal advice**. It may be incorrect or outdated.
Always consult a licensed lawyer for serious matters.

### Links
- 🐙 [GitHub Repository](https://github.com/jishanmujawar33/LexPal)
- 🇮🇳 [India Code (official law texts)](https://indiacode.nic.in)
- ⚖️ [Bar Council of India](https://www.barcouncilofindia.org)

</div>
""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#475569;font-size:0.82rem;padding:0.5rem 0;">'
    '⚖️ LexPal · Legal Information Only · Not a Lawyer · Open Source · '
    '<a href="https://github.com/jishanmujawar33/LexPal" style="color:#818cf8;">GitHub</a>'
    '</div>',
    unsafe_allow_html=True,
)
