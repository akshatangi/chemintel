"""Reaction PRECEDENT retrieval. It finds recorded reactions similar to a query. It does NOT predict whether a reaction will work,
its yield or its safety. Load your own records (e.g. exported from the Open Reaction Database, CC BY-SA 4.0, check terms before redistributing).
Record format, one JSON object per line: {"id","reactants":"...","products":"...","conditions":"...","yield":"...","source":"..."}"""
import json
from .bm25 import BM25Index

DISCLAIMER = ("Precedents only: these are recorded reactions with similar words. They are not a prediction of feasibility, "
              "yield or safety. Check the cited source and use chemical judgement.")


def load_records(path):
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def build_index(records):
    idx = BM25Index()
    for r in records:
        text = f"reactants {r.get('reactants','')} products {r.get('products','')} conditions {r.get('conditions','')}"
        idx.add(text, str(r.get("id", "")), r.get("source", ""))
    idx.finalize()
    idx.records = {str(r.get("id", "")): r for r in records}
    return idx


def find_precedents(idx, query, k=3):
    hits = idx.search(query, k=k)
    return [{"record": idx.records[h["doc_id"]], "score": h["score"], "note": DISCLAIMER} for h in hits]
