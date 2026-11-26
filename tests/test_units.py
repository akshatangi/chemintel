"""Unit tests (no network, no model). Run: python -m pytest -q tests  or  python tests/test_units.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chemintel.bm25 import BM25Index, chunk_text, tokenize
from chemintel.decontam import Decontaminator, CANARY_GUID
from chemintel.bench import build_prompt, pick_answer, letter_token_ids, wilson, difficulty_of


def test_chunk_covers_text():
    t = "word " * 1000
    ch = chunk_text(t, size=200, overlap=40)
    assert len(ch) > 5 and all(len(c) <= 200 for c in ch)
    assert ch[0].startswith("word") and ch[-1].endswith("word")


def test_bm25_ranks_relevant_first():
    i = BM25Index()
    i.add("sodium chloride dissolves in water and raises boiling point", "a")
    i.add("the mitochondria is the powerhouse of the cell", "b")
    i.add("Le Chatelier principle equilibrium shift", "c")
    i.finalize()
    assert i.search("boiling point of salt water", k=1)[0]["doc_id"] == "a"
    assert i.search("zzzz unknown", k=1) == []


def test_decontam_drops_overlap_and_canary():
    q = "What is the number of peaks in the 1H NMR spectrum of a molecule with the given structure shown here"
    d = Decontaminator([q], n=10)
    assert d.is_contaminated("intro text " + q + " more text")
    assert d.is_contaminated("anything " + CANARY_GUID)
    assert not d.is_contaminated("completely different chemistry text about acids and bases and their reactions in water")
    assert d.stats()["docs_dropped"] == 2


def test_prompt_and_pick():
    item = {"question": "Q?", "options": ["x", "y", "z"], "scores": [0, 1, 0]}
    p = build_prompt(item, passages=[{"text": "ctx"}])
    assert "A. x" in p and "C. z" in p and p.endswith("Answer:") and "[1] ctx" in p
    best, vals = pick_answer([0.0, -5.0, -1.0, -9.0], [[1], [2], [3]])
    assert best == 1  # logprob -1.0 > -5.0 > -9.0 ... option index 1 uses token 2
    lo, hi = wilson(50, 100)
    assert 0.39 < lo < 0.41 and 0.59 < hi < 0.61
    assert difficulty_of(["x", "difficulty-advanced"]) == "advanced" and difficulty_of([]) == "unlabelled"




def test_pdf_pages_are_indexed_with_page_numbers():
    import io
    from fpdf import FPDF
    from chemintel.pdfqa import build_pdf_index, search_many
    pdf = FPDF(); pdf.set_font("Helvetica", size=12)
    pdf.add_page(); pdf.multi_cell(0, 8, "Flash point: 38 C closed cup. Density 1.35 g/cm3.")
    pdf.add_page(); pdf.multi_cell(0, 8, "Shelf life is 12 months when stored closed.")
    ix, info = build_pdf_index(bytes(pdf.output()), "t.pdf")
    assert info["pages"] == 2 and info["pages_with_text"] == 2
    assert search_many([ix], "what is the shelf life", k=1)[0]["source"] == "t.pdf, page 2"
    assert search_many([ix], "zzzz", k=1) == []


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
