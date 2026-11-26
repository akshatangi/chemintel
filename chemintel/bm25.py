"""Small pure-python BM25 over text chunks. No dependencies, fine for a few hundred thousand chunks."""
import math, re, json, pickle
from collections import Counter, defaultdict

_tok = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def tokenize(text):
    return [t.lower() for t in _tok.findall(text)]


def chunk_text(text, size=900, overlap=150):
    """Split on characters, preferring a space near the boundary. Returns list of strings."""
    text = " ".join(text.split())
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):
            k = text.rfind(" ", i + size // 2, j)
            if k > 0:
                j = k
        out.append(text[i:j])
        if j >= len(text):
            break
        i = max(j - overlap, i + 1)
    return out


class BM25Index:
    def __init__(self, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.chunks = []      # list of {"text", "doc_id", "source"}
        self.postings = defaultdict(list)  # term -> [(chunk_id, tf)]
        self.doc_len = []
        self.avg_len = 0.0

    def add(self, text, doc_id, source=""):
        cid = len(self.chunks)
        toks = tokenize(text)
        self.chunks.append({"text": text, "doc_id": doc_id, "source": source})
        self.doc_len.append(len(toks))
        for term, tf in Counter(toks).items():
            self.postings[term].append((cid, tf))

    def finalize(self):
        self.avg_len = sum(self.doc_len) / max(1, len(self.doc_len))

    def search(self, query, k=3):
        n = len(self.chunks)
        scores = defaultdict(float)
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for cid, tf in plist:
                dl = self.doc_len[cid]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_len)
                scores[cid] += idf * tf * (self.k1 + 1) / denom
        top = sorted(scores.items(), key=lambda x: -x[1])[:k]
        return [dict(self.chunks[c], score=s, chunk_id=c) for c, s in top]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        o = cls()
        with open(path, "rb") as f:
            o.__dict__.update(pickle.load(f))
        return o
