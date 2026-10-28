"""
utils/extractor.py
Text extraction from PDF, DOCX, and TXT files.
Falls back to OCR for scanned/image-based PDFs.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text(file_path: str | Path) -> str:
    """
    Auto-detect file type and extract text.
    Returns extracted text or raises ValueError for unsupported types.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(path)
    elif suffix == ".docx":
        return extract_docx(path)
    elif suffix == ".txt":
        return extract_txt(path)
    elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
        from utils.ocr import ocr_image
        return ocr_image(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def extract_pdf(path: Path) -> str:
    """
    Extract text from a PDF.
    Uses pypdf for digital text; falls back page-by-page to OCR for scanned pages.
    """
    import pypdf
    from utils.ocr import ocr_pdf_page

    text_parts = []
    try:
        reader = pypdf.PdfReader(str(path))
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            # Heuristic: if fewer than 30 chars extracted, treat as scanned
            if len(page_text.strip()) < 30:
                logger.info(f"Page {page_num + 1} appears scanned, running OCR…")
                ocr_text = ocr_pdf_page(str(path), page_num)
                text_parts.append(f"[Page {page_num + 1}]\n{ocr_text}")
            else:
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        raise

    return "\n\n".join(text_parts)


def extract_docx(path: Path) -> str:
    """Extract text from a .docx file using python-docx."""
    try:
        import docx
        doc = docx.Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        raise


def extract_txt(path: Path) -> str:
    """Read a plain text file, trying UTF-8 then latin-1."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")
