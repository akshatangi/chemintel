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




def test_datasheet_regex_fields_and_provenance():
    from chemintel.datasheet import extract_regex
    ch = [{"text": "Density at 25 C: 1.35 g/cm3\nFlash point: 38 C (closed cup)\nPot life: 4 hours at 23 C\nH226 Flammable liquid and vapour. CAS 64-17-5 and 64-17-6 listed. Shelf life 12 months.", "source": "x.pdf, page 1"}]
    got = {(d["field"], d["value"], d["unit"]) for d in extract_regex(ch)}
    assert ("density", "1.35", "g/cm3") in got and ("flash point", "38", "C") in got
    assert ("pot life", "4", "hours") in got and ("shelf life", "12", "months") in got
    assert any(f == "hazard statement" and v == "H226" for f, v, _ in got)
    cas = {v for f, v, _ in got if f.startswith("CAS")}
    assert cas == {"64-17-5"}               # 64-17-6 fails the checksum
    assert all(d["source"] == "x.pdf, page 1" and d["method"] == "regex" for d in extract_regex(ch))
    assert extract_regex([{"text": "nothing here", "source": "s"}]) == []


def test_datasheet_llm_values_must_appear_in_text():
    from chemintel.datasheet import extract_llm
    class Fake:
        def generate(self, p, **k):
            return '[{"field":"flash point","value":"38","unit":"C"},{"field":"boiling point","value":"99","unit":"C"}]'
    kept, rej = extract_llm(Fake(), [{"text": "Flash point: 38 C", "source": "s"}])
    assert [k["value"] for k in kept] == ["38"] and rej == 1 and "not validated" in kept[0]["method"]


def test_reaction_precedents_are_labelled_and_ranked():
    import os
    from chemintel.reactions import load_records, build_index, find_precedents
    p = os.path.join(os.path.dirname(__file__), "..", "examples", "reactions_SYNTHETIC.jsonl")
    idx = build_index(load_records(p))
    r = find_precedents(idx, "acetic acid and ethanol esterification", k=1)
    assert r[0]["record"]["id"] == "syn-1" and "not a prediction" in r[0]["note"]




def test_reaction_no_match_returns_nothing():
    import os
    from chemintel.reactions import load_records, build_index, find_precedents
    p = os.path.join(os.path.dirname(__file__), "..", "examples", "reactions_SYNTHETIC.jsonl")
    idx = build_index(load_records(p))
    assert find_precedents(idx, "zzzz qqqq", k=3) == []
    assert find_precedents(idx, "suzuki coupling boronic acid palladium", k=3) == []   # shares no word with the records
    assert find_precedents(idx, "the and of", k=3) == []


def test_ord_importer_roundtrip(tmp_path=None):
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        from ord_schema.proto import reaction_pb2
    except Exception:
        print("skip (pyarrow/ord-schema not installed)"); return
    import json, subprocess, sys, tempfile, os
    r = reaction_pb2.Reaction()
    r.identifiers.add(value="9.9 [Test] Fake amination")
    c = r.inputs["amine"].components.add(); c.identifiers.add(value="CCN", type=reaction_pb2.CompoundIdentifier.SMILES)
    o = r.outcomes.add(); pr = o.products.add(); pr.identifiers.add(value="CCNC", type=reaction_pb2.CompoundIdentifier.SMILES)
    m = pr.measurements.add(type=reaction_pb2.ProductMeasurement.YIELD); m.percentage.value = 42.0
    d = tempfile.mkdtemp(); f = os.path.join(d, "t.parquet")
    pq.write_table(pa.table({"reaction_id": ["ord-test"], "reaction": [r.SerializeToString()]}), f)
    out = os.path.join(d, "o.jsonl")
    root = os.path.join(os.path.dirname(__file__), "..")
    subprocess.check_call([sys.executable, os.path.join(root, "scripts", "07_import_ord.py"), "--dataset", f, "--out", out, "--limit", "5"], stdout=subprocess.DEVNULL)
    rec = json.loads(open(out).read())
    assert rec["id"] == "ord-test" and "CCN" in rec["reactants"] and rec["products"] == "CCNC"
    assert rec["yield"].startswith("42.0") and "CC BY-SA 4.0" in rec["source"]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
