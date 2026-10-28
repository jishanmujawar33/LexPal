"""
analyze.py  –  LexPal Document Analysis Router
=================================================
Endpoint:
  POST /analyze  – upload a legal document (PDF/DOCX/JPG/PNG/TXT)
                   and receive a structured analysis with next steps.

Mount this router in main.py with:
  app.include_router(analyze_router)
"""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from config import (
    ANALYZE_SYSTEM_PROMPT,
    DEFAULT_LANGUAGE,
    DISCLAIMER,
    SUPPORTED_LANGUAGES,
    TOP_K,
)

logger = logging.getLogger("lexpal.analyze")

analyze_router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
MAX_FILE_SIZE_MB   = 10


class AnalyzeResponse(BaseModel):
    extracted_text: str
    summary: str
    analysis: str          # full LLM output
    sources: list[dict]
    language: str
    disclaimer: str
    ocr_used: bool


@analyze_router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_document(
    file: UploadFile = File(...),
    language: str    = Form(DEFAULT_LANGUAGE),
):
    """
    Accept a legal document, extract its text, retrieve relevant law chunks,
    and return a structured analysis with next steps.
    """
    # ── Validate language ───────────────────────────────────────────────────
    if language not in SUPPORTED_LANGUAGES:
        language = DEFAULT_LANGUAGE

    # ── Validate file type ──────────────────────────────────────────────────
    suffix = Path(file.filename or "file.txt").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # ── Read file into memory (never persisted after request) ───────────────
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum is {MAX_FILE_SIZE_MB} MB.",
        )

    # ── Write to temp file (needed by extraction libs) ──────────────────────
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    ocr_used = False
    try:
        # ── Extract text ────────────────────────────────────────────────────
        from utils.extractor import extract_text
        from utils.ocr import ocr_image

        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
            extracted_text = ocr_image(tmp_path)
            ocr_used = True
        else:
            extracted_text = extract_text(tmp_path)
            # Check if OCR was triggered inside extract_pdf
            ocr_used = "[OCR" in extracted_text

        if not extracted_text.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not extract any text from the uploaded file.",
            )

        # ── Retrieve relevant law chunks ─────────────────────────────────────
        # Import retrieval helpers from main module
        from main import retrieve_chunks, call_ollama, build_context

        # Use first 1000 chars of doc as query for retrieval
        query_text = extracted_text[:1000]
        chunks = retrieve_chunks(query_text, TOP_K)
        context = build_context(chunks) if chunks else "No relevant laws found in database."

        # ── Call LLM ─────────────────────────────────────────────────────────
        system = ANALYZE_SYSTEM_PROMPT.format(
            language=language,
            document_text=extracted_text[:3000],   # cap to avoid context overflow
            context=context,
        )
        analysis = call_ollama(system, "Analyze this document and provide next steps.")

        # ── Build source list ────────────────────────────────────────────────
        sources = []
        for c in chunks:
            m = c["metadata"]
            sources.append({
                "act":     m.get("act", "Unknown"),
                "section": m.get("section", "Unknown"),
                "page":    m.get("page", "?"),
                "snippet": c["text"][:200],
            })

        return AnalyzeResponse(
            extracted_text=extracted_text[:5000],  # preview cap
            summary=analysis.split("\n")[0] if analysis else "",
            analysis=analysis,
            sources=sources,
            language=language,
            disclaimer=DISCLAIMER,
            ocr_used=ocr_used,
        )

    finally:
        # ── Always delete temp file ──────────────────────────────────────────
        tmp_path.unlink(missing_ok=True)
