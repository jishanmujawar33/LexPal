"""
utils/ocr.py
Tesseract OCR wrapper for images and scanned PDF pages.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Default Tesseract language string (English + Hindi + Marathi)
DEFAULT_LANG = "eng+hin+mar"


def ocr_image(image_path: str | Path, lang: str = DEFAULT_LANG) -> str:
    """
    Run Tesseract OCR on an image file.
    Returns extracted text string.
    """
    try:
        import pytesseract
        from PIL import Image

        img = Image.open(str(image_path))
        # Upscale small images for better accuracy
        if img.width < 1000:
            scale = 1000 / img.width
            img = img.resize(
                (int(img.width * scale), int(img.height * scale)),
                Image.LANCZOS,
            )
        text = pytesseract.image_to_string(img, lang=lang)
        return text.strip()
    except Exception as e:
        logger.error(f"OCR failed on {image_path}: {e}")
        return f"[OCR failed: {e}]"


def ocr_pdf_page(pdf_path: str, page_num: int, lang: str = DEFAULT_LANG) -> str:
    """
    Convert a single PDF page to an image and OCR it.
    Requires poppler (pdf2image).
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract

        images = convert_from_path(
            pdf_path,
            first_page=page_num + 1,
            last_page=page_num + 1,
            dpi=300,
        )
        if not images:
            return ""
        text = pytesseract.image_to_string(images[0], lang=lang)
        return text.strip()
    except Exception as e:
        logger.error(f"OCR PDF page {page_num} failed: {e}")
        return f"[OCR failed on page {page_num + 1}: {e}]"
