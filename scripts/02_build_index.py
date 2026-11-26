"""Chunk the prepared corpus and build a BM25 index used by the retrieval (RAG) condition.
This is a standalone baseline that retrieves ChemPile passages. It is NOT the project's existing RAG."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chemintel.bm25 import BM25Index, chunk_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/train.jsonl")
    ap.add_argument("--out", default="data/bm25.pkl")
    ap.add_argument("--max-docs", type=int, default=0, help="0 = all")
    ap.add_argument("--chunk-chars", type=int, default=900)
    a = ap.parse_args()
    idx = BM25Index()
    n = 0
    for line in open(a.corpus):
        r = json.loads(line)
        for c in chunk_text(r["text"], size=a.chunk_chars):
            idx.add(c, r["id"], r.get("source", ""))
        n += 1
        if a.max_docs and n >= a.max_docs:
            break
    idx.finalize()
    idx.save(a.out)
    print(json.dumps({"docs": n, "chunks": len(idx.chunks), "index": a.out}))


if __name__ == "__main__":
    main()
