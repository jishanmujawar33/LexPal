# ⚖️ LexPal – AI-Powered Legal Information Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **LexPal provides legal information, not legal advice. Always consult a licensed lawyer for serious matters.**

LexPal helps ordinary people understand their legal rights in plain English, Hindi, or Marathi. Ask a legal question or upload a document (notice, agreement, court order, FIR copy, or photo) and receive:
- A clear explanation in simple language
- The relevant laws with Act name, Section, and page number
- Practical next steps

All answers are built from **real law texts stored in a local database** using Retrieval-Augmented Generation (RAG) — not from the model's memory. Every claim cites its source.

---

## 🗂️ Laws Covered (MVP)
| Act | Year |
|-----|------|
| Consumer Protection Act | 2019 |
| Negotiable Instruments Act | 1881 |
| Information Technology Act | 2000 |
| Maharashtra Rent Control Act | 1999 |
| Right to Information Act | 2005 |

---

## 🏗️ Architecture

```
User Query / Document
        │
        ▼
[Streamlit Frontend]
        │  HTTP
        ▼
[FastAPI Backend]
  /ask   /analyze   /ingest
        │
        ├── Embed query (sentence-transformers)
        ├── Retrieve top-5 law chunks (ChromaDB)
        ├── Build strict RAG prompt
        └── Call Ollama LLM (Llama 3.1 8B)
                │
                └── Cited answer + sources
```

---

## ⚡ Quick Setup

### 1. System Prerequisites (macOS)

```bash
# Install Homebrew (if not present)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Tesseract OCR with language packs
brew install tesseract tesseract-lang

# Install Poppler (for pdf2image)
brew install poppler

# Install Ollama
brew install ollama
```

### 2. Start Ollama & Pull Model

```bash
# Start the Ollama server (keep this running in a terminal)
ollama serve

# In another terminal, pull the LLM
ollama pull llama3.1:8b
```

> 💡 **Weak laptop?** Use `ollama pull llama3.2:3b` instead and set `OLLAMA_MODEL=llama3.2:3b` in `.env`.

### 3. Clone & Install Python Packages

```bash
git clone https://github.com/jishanmujawar33/LexPal.git
cd LexPal

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Download Law PDFs

Place official bare-act PDFs in the `data/` folder. Recommended sources:

| Act | Download Link |
|-----|---------------|
| Consumer Protection Act 2019 | [consumeraffairs.nic.in](https://consumeraffairs.nic.in/sites/default/files/file-uploads/acts-and-rules/Consumer%20Protection%20Act%202019.pdf) |
| Negotiable Instruments Act 1881 | [indiacode.nic.in](https://indiacode.nic.in/bitstream/123456789/2051/1/188126.pdf) |
| IT Act 2000 | [indiacode.nic.in](https://indiacode.nic.in/bitstream/123456789/1999/3/200021.pdf) |
| RTI Act 2005 | [rti.gov.in](https://rti.gov.in/rti-act.pdf) |
| Maharashtra Rent Control Act 1999 | [maharerait.mahaonline.gov.in](https://maharerait.mahaonline.gov.in/images/pdf/Maharashtra_Rent_Control_act_1999.pdf) |

```bash
# Or download via curl (example)
curl -L "https://rti.gov.in/rti-act.pdf" -o data/rti.pdf
```

### 5. Ingest Laws into ChromaDB

```bash
python ingest.py
```

This reads every PDF in `data/`, chunks it section-by-section, embeds each chunk, and stores it in `legal_db/` (ChromaDB). Run once; re-run only when laws change.

### 6. Start the Backend

```bash
uvicorn main:app --reload --port 8000
```

Verify: open http://localhost:8000/health

### 7. Start the Frontend

```bash
# In a new terminal (with venv active)
streamlit run frontend.py
```

Open: http://localhost:8501

---

## 📂 Project Structure

```
LexPal/
├── main.py               ← FastAPI: /health, /ingest, /ask
├── analyze.py            ← FastAPI router: /analyze (document upload)
├── ingest.py             ← CLI ingestion pipeline
├── frontend.py           ← Streamlit UI
├── config.py             ← All constants and prompts
├── utils/
│   ├── extractor.py      ← PDF / DOCX / TXT extraction
│   ├── ocr.py            ← Tesseract OCR wrapper
│   └── chunker.py        ← Section-aware text splitter
├── data/                 ← Put law PDFs here
├── legal_db/             ← ChromaDB storage (auto-created)
├── tests/
│   ├── test_questions.json  ← 30 Q&A pairs
│   └── eval.py              ← Evaluation script
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🧪 Running the Evaluation

```bash
# Make sure backend is running first
python tests/eval.py
```

Measures:
- **Keyword accuracy** – does the answer contain expected terms?
- **Citation accuracy** – is the expected section cited in sources?
- **Hallucination rate** – does the model invent section numbers?

Target: ≥ 80% keyword accuracy, 0 hallucinated sections.

---

## 🌐 Optional: Free Cloud Deployment

### Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select `frontend.py`.
3. Note: Ollama won't run on Streamlit Cloud. Point `OLLAMA_BASE_URL` to a remote instance or swap to a free API.

### Hugging Face Spaces (Docker)
See `Dockerfile` in the repo root.

---

## 🔒 Privacy
- Uploaded documents are processed **in memory only** — never written to disk permanently.
- No queries or personal information are stored or logged.

---

## ⚠️ Disclaimer
LexPal is a **legal information tool, not a lawyer**. It may be incorrect or outdated. For criminal charges, court deadlines, or claims above ₹20 lakh, consult a licensed lawyer via the [Bar Council of India](https://www.barcouncilofindia.org).

---

## 📄 License
MIT License – see [LICENSE](LICENSE).
