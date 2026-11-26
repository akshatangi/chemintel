"""Datasheet field extraction. Two methods, always labelled:
  method="regex"  : deterministic patterns, value and unit copied from the text, with the page. HEURISTIC, not validated by a chemist.
  method="llm"    : a model proposes JSON; a value is kept only if its digits/text literally appear in the cited source passage.
                    Passing that check means "present in the text", not "correct".
Missing fields are returned as absent, never guessed. Scanned PDFs need OCR first (not included)."""
import json, re

NUM = r"(-?\d+(?:[.,]\d+)?(?:\s*(?:-|to)\s*-?\d+(?:[.,]\d+)?)?)"
PATTERNS = {
    "density": (r"density[^0-9\n]{0,40}?(?:at\s*\d+\s*(?:°|deg)?\s*C[^0-9\n]{0,10})?" + NUM + r"\s*(g/cm3|g/cm³|g/ml|kg/m3|kg/m³|g/cc)", "density"),
    "flash_point": (r"flash\s*point[^0-9\n-]{0,30}" + NUM + r"\s*(?:°\s*)?(C|F)\b", "flash point"),
    "boiling_point": (r"boiling\s*point[^0-9\n-]{0,30}" + NUM + r"\s*(?:°\s*)?(C|F)\b", "boiling point"),
    "melting_point": (r"melting\s*point[^0-9\n-]{0,30}" + NUM + r"\s*(?:°\s*)?(C|F)\b", "melting point"),
    "autoignition": (r"auto-?\s*ignition\s*temp\w*[^0-9\n-]{0,30}" + NUM + r"\s*(?:°\s*)?(C|F)\b", "auto-ignition temperature"),
    "pot_life": (r"pot\s*life[^0-9\n]{0,30}" + NUM + r"\s*(hours?|hrs?|h|minutes?|min)\b", "pot life"),
    "shelf_life": (r"shelf\s*life[^0-9\n]{0,30}" + NUM + r"\s*(months?|years?|days?)\b", "shelf life"),
    "ph": (r"\bpH\b[^0-9\n]{0,20}" + NUM, "pH"),
}
H_RE = re.compile(r"\b(H\d{3}[A-Za-z]?)\b\s*:?\s*([^.\n]{0,80})")
CAS_RE = re.compile(r"\b(\d{2,7}-\d{2}-\d)\b")


def _cas_ok(s):
    d = s.replace("-", "")
    return sum(int(c) * i for i, c in enumerate(reversed(d[:-1]), 1)) % 10 == int(d[-1])


def extract_regex(chunks):
    """chunks: list of {"text","source"} (e.g. a PDF index's .chunks). Returns a list of field dicts."""
    out, seen = [], set()
    for ch in chunks:
        t = ch["text"]
        for key, (pat, label) in PATTERNS.items():
            for m in re.finditer(pat, t, re.I):
                val = m.group(1).strip(); unit = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else ""
                k = (key, val, unit)
                if k in seen: continue
                seen.add(k)
                out.append({"field": label, "value": val, "unit": unit, "source": ch["source"], "evidence": t[max(0, m.start()-20):m.end()+20].replace("\n", " "), "method": "regex"})
        for m in H_RE.finditer(t):
            k = ("H", m.group(1))
            if k in seen: continue
            seen.add(k)
            out.append({"field": "hazard statement", "value": m.group(1), "unit": "", "source": ch["source"], "evidence": m.group(0).strip(), "method": "regex"})
        for m in CAS_RE.finditer(t):
            if not _cas_ok(m.group(1)): continue
            k = ("CAS", m.group(1))
            if k in seen: continue
            seen.add(k)
            out.append({"field": "CAS number (checksum valid)", "value": m.group(1), "unit": "", "source": ch["source"], "evidence": t[max(0, m.start()-20):m.end()+10].replace("\n", " "), "method": "regex"})
    return out


LLM_PROMPT = ("Extract fields from the datasheet text. Reply with JSON only: a list of objects with keys "
              '"field","value","unit". Use only values written in the text. Omit anything not stated.\n\nTEXT:\n')


def extract_llm(generator, chunks, max_chars=1500):
    """Ask a model per passage; keep only items whose value appears literally in that passage. Untested with a real model here."""
    kept, rejected = [], 0
    for ch in chunks:
        raw = generator.generate(LLM_PROMPT + ch["text"][:max_chars], max_new_tokens=300, temperature=0.0)
        m = re.search(r"\[.*\]", raw, re.S)
        try:
            items = json.loads(m.group(0)) if m else []
        except Exception:
            items = []
        for it in items if isinstance(items, list) else []:
            if not isinstance(it, dict) or "value" not in it: rejected += 1; continue
            v = str(it["value"]).strip()
            if v and v in ch["text"]:
                kept.append({"field": str(it.get("field", "")), "value": v, "unit": str(it.get("unit", "")), "source": ch["source"], "evidence": "", "method": "llm (value found in text; correctness not validated)"})
            else:
                rejected += 1
    return kept, rejected
