"""
PDF -> raw text. Uses pdfplumber (better for real-world PDFs) with pypdf fallback.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

log = logging.getLogger("pdf_extractor")


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> str:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Try pdfplumber first (handles layout better)
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
        text = "\n".join(text_parts)
        if text.strip():
            log.info(f"pdfplumber extracted {len(text)} chars from {pdf_path.name}")
            return text
    except Exception as e:
        log.warning(f"pdfplumber failed ({e}), falling back to pypdf")

    # Fallback: pypdf
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    log.info(f"pypdf extracted {len(text)} chars from {pdf_path.name}")
    return text


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """For API uploads."""
    import io
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        if text.strip():
            return text
    except Exception as e:
        log.warning(f"pdfplumber (bytes) failed: {e}")

    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join((page.extract_text() or "") for page in reader.pages)
