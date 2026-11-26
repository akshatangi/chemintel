"""PDF to searchable passages with page numbers (pypdf, BSD licence). Text-based PDFs only; scanned PDFs need OCR (not included)."""
import io
from .bm25 import BM25Index, chunk_text


def _page_text(page):
    """Layout mode keeps table rows (label and value on one line) together; falls back to default extraction on old pypdf."""
    try:
        t = page.extract_text(extraction_mode="layout") or ""
    except TypeError:
        t = page.extract_text() or ""
    import re
    return re.sub(r"[ \t]{2,}", "  ", t).strip()


def build_pdf_index(data: bytes, name="uploaded.pdf", chunk_chars=600):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    idx = BM25Index()
    pages_with_text = 0
    for pno, page in enumerate(reader.pages, 1):
        text = _page_text(page)
        if not text:
            continue
        pages_with_text += 1
        for c in chunk_text(text, size=chunk_chars, overlap=80):
            idx.add(c, f"{name}:p{pno}", f"{name}, page {pno}")
    idx.finalize()
    return idx, {"pages": len(reader.pages), "pages_with_text": pages_with_text, "passages": len(idx.chunks)}


def search_many(indexes, query, k=3):
    """Search several indexes, return the top-k passages overall (scores are comparable only roughly across indexes)."""
    hits = []
    for ix in indexes:
        hits.extend(ix.search(query, k=k))
    return sorted(hits, key=lambda x: -x["score"])[:k]
