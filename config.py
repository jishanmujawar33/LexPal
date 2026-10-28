"""
LexPal – Central Configuration
All tuneable constants live here; never hard-code these elsewhere.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
DATA_DIR   = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "legal_db"

# ── Ollama / LLM ──────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# ── ChromaDB ──────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "legal_texts"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500   # tokens (approx characters / 4)
CHUNK_OVERLAP = 50

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5             # number of chunks to retrieve per query

# ── Languages ─────────────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = ["English", "Hindi", "Marathi"]
DEFAULT_LANGUAGE    = "English"

# Tesseract language codes
TESSERACT_LANG_MAP = {
    "English": "eng",
    "Hindi":   "hin",
    "Marathi": "mar",
}

# ── Acts covered (for display / metadata) ─────────────────────────────────────
COVERED_ACTS = [
    "Consumer Protection Act 2019",
    "Negotiable Instruments Act 1881",
    "Information Technology Act 2000",
    "Maharashtra Rent Control Act 1999",
    "Right to Information Act 2005",
]

# ── Disclaimer ────────────────────────────────────────────────────────────────
DISCLAIMER = (
    "⚠️ DISCLAIMER: LexPal provides legal information, NOT legal advice. "
    "For serious matters (criminal charges, court deadlines, large financial claims) "
    "please consult a licensed lawyer."
)

# ── Strict RAG System Prompt ──────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are LexPal, a legal information assistant for India.
Your ONLY job is to explain legal concepts in plain language based on the provided context.

STRICT RULES:
1. Answer ONLY using the provided context. Do NOT use general knowledge.
2. If the context does not contain the answer, say exactly:
   "I don't have enough information on this in my database. Please consult a licensed lawyer."
3. Always cite: Act name, Section number, and page number for EVERY fact you state.
4. Keep the explanation simple — assume the user has no legal background.
5. End every answer with the disclaimer:
   "⚠️ This is legal information, not legal advice. Consult a licensed lawyer for your specific situation."
6. Respond in {language}.
7. Never invent section numbers, dates, or penalties.

Context:
{context}
"""

ANALYZE_SYSTEM_PROMPT = """You are LexPal, a legal document analyst for India.
Analyze the provided document text and relevant law context.
Return a structured analysis with:
1. **Document Summary**: What type of document is this and what does it say in simple terms?
2. **Relevant Laws**: Which laws apply? Cite Act name and Section.
3. **Your Rights**: What rights does the person have in this situation?
4. **Next Steps**: Practical, numbered action steps the person should take.
5. **Should you consult a lawyer?**: Give a clear YES/NO with reason.

STRICT RULES:
- Only refer to laws found in the provided context.
- If a law is not in the context, do not mention it.
- Never predict case outcomes.
- Respond in {language}.
- End with: "⚠️ This is legal information, not legal advice."

Document text:
{document_text}

Relevant law context:
{context}
"""
