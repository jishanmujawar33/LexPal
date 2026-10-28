"""
main.py  –  LexPal FastAPI Backend
====================================
Endpoints:
  GET  /health        – liveness check
  POST /ingest        – run ingestion pipeline on data/ folder
  POST /ask           – answer a legal question using RAG
"""

import logging
import subprocess
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    CHROMA_COLLECTION,
    CHROMA_DIR,
    COVERED_ACTS,
    DEFAULT_LANGUAGE,
    DISCLAIMER,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RAG_SYSTEM_PROMPT,
    SUPPORTED_LANGUAGES,
    TOP_K,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("lexpal.main")

# ── Globals (loaded once at startup) ─────────────────────────────────────────
embed_model    = None
chroma_collection = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load embedding model and ChromaDB at startup."""
    global embed_model, chroma_collection

    logger.info("Loading embedding model…")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer(EMBED_MODEL)

    logger.info("Connecting to ChromaDB…")
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info(f"ChromaDB ready. Collection has {chroma_collection.count()} chunks.")
    yield
    logger.info("Shutting down LexPal backend.")


app = FastAPI(
    title="LexPal API",
    description="Free AI-powered legal information assistant for India",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount document analysis router
from analyze import analyze_router  # noqa: E402
app.include_router(analyze_router)

# ── Pydantic models ───────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)
    language: str = Field(DEFAULT_LANGUAGE)

class SourceCitation(BaseModel):
    act: str
    section: str
    page: str
    snippet: str

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    language: str
    disclaimer: str

class IngestResponse(BaseModel):
    status: str
    message: str

# ── Helpers ───────────────────────────────────────────────────────────────────

def embed_query(text: str) -> list[float]:
    return embed_model.encode([text]).tolist()[0]


def retrieve_chunks(query: str, n: int = TOP_K) -> list[dict]:
    """Retrieve top-k relevant law chunks from ChromaDB."""
    if chroma_collection.count() == 0:
        return []
    q_emb = embed_query(query)
    results = chroma_collection.query(
        query_embeddings=[q_emb],
        n_results=min(n, chroma_collection.count()),
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


def call_llm(system_prompt: str, user_message: str) -> str:
    """
    Call the configured LLM backend.
    - LLM_BACKEND=ollama (default): local Ollama server
    - LLM_BACKEND=groq: Groq cloud API (free tier, fast)
    """
    from config import LLM_BACKEND
    if LLM_BACKEND == "groq":
        return _call_groq(system_prompt, user_message)
    return _call_ollama(system_prompt, user_message)


def _call_groq(system_prompt: str, user_message: str) -> str:
    """Call Groq free cloud API (Llama 3 hosted by Groq)."""
    import httpx
    from config import GROQ_API_KEY, GROQ_MODEL

    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY environment variable is not set. Get a free key at https://console.groq.com",
        )
    try:
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                "temperature": 0.1,
                "max_tokens": 1024,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot reach Groq API. Check your internet connection.")
    except Exception as e:
        logger.error(f"Groq error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _call_ollama(system_prompt: str, user_message: str) -> str:
    """Call the local Ollama LLM via its HTTP API."""
    import httpx

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "stream": False,
    }
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama is not running. Start it with: ollama serve\n"
                "Then pull the model: ollama pull llama3.1:8b\n"
                "Or set LLM_BACKEND=groq and GROQ_API_KEY for cloud deployment."
            ),
        )
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Keep backward-compatible alias
def call_ollama(system_prompt: str, user_message: str) -> str:
    return call_llm(system_prompt, user_message)



def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        parts.append(
            f"[Source {i}] {m.get('act', '?')} | {m.get('section', '?')} | Page {m.get('page', '?')}\n"
            f"{c['text']}"
        )
    return "\n\n---\n\n".join(parts)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    db_count = chroma_collection.count() if chroma_collection else 0
    return {
        "status": "ok",
        "db_chunks": db_count,
        "covered_acts": COVERED_ACTS,
        "model": OLLAMA_MODEL,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """Trigger the ingestion pipeline on the data/ directory."""
    logger.info("Ingest endpoint triggered.")
    try:
        result = subprocess.run(
            [sys.executable, "ingest.py"],
            capture_output=True,
            text=True,
            cwd=str(CHROMA_DIR.parent),
        )
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)
        return IngestResponse(
            status="success",
            message=result.stdout.strip() or "Ingestion complete.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Answer a legal question using RAG."""
    language = req.language if req.language in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE

    # 1. Retrieve relevant chunks
    chunks = retrieve_chunks(req.question, TOP_K)

    if not chunks:
        return AskResponse(
            answer=(
                "My legal database is currently empty. "
                "Please ask the administrator to run the ingestion pipeline first. "
                "In the meantime, please consult a licensed lawyer."
            ),
            sources=[],
            language=language,
            disclaimer=DISCLAIMER,
        )

    # 2. Build context string
    context = build_context(chunks)

    # 3. Format system prompt
    system = RAG_SYSTEM_PROMPT.format(language=language, context=context)

    # 4. Call LLM
    answer = call_ollama(system, req.question)

    # 5. Build source citations
    sources = []
    for c in chunks:
        m = c["metadata"]
        sources.append(SourceCitation(
            act=m.get("act", "Unknown"),
            section=m.get("section", "Unknown"),
            page=m.get("page", "?"),
            snippet=c["text"][:200] + ("…" if len(c["text"]) > 200 else ""),
        ))

    return AskResponse(
        answer=answer,
        sources=sources,
        language=language,
        disclaimer=DISCLAIMER,
    )
