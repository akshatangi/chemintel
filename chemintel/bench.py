"""ChemBench loading, prompt building and scoring.

Dataset: jablonkagroup/ChemBench on Hugging Face (MIT per its metadata). Each row has one example with
`input` and either `target_scores` (multiple choice, JSON string option->score) or `target` (numeric / string).
This code evaluates the multiple-choice questions only; numeric ones are skipped and counted.

ChemBench data carries a canary string: it must never be used for training. We only evaluate on it.
"""
import json, math, random

SUBFIELDS = ["analytical_chemistry", "chemical_preference", "general_chemistry", "inorganic_chemistry",
             "materials_science", "organic_chemistry", "physical_chemistry", "technical_chemistry",
             "toxicity_and_safety"]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def difficulty_of(keywords):
    ks = [k.lower() for k in (keywords or [])]
    if "difficulty-basic" in ks:
        return "basic"
    if "difficulty-advanced" in ks:
        return "advanced"
    return "unlabelled"


def load_chembench(subfields=None, seed=0, limit=None):
    from datasets import load_dataset
    items, skipped = [], 0
    for cfg in (subfields or SUBFIELDS):
        ds = load_dataset("jablonkagroup/ChemBench", cfg, split="train")
        for r in ds:
            ex = r["examples"][0]
            ts = ex.get("target_scores")
            if not ts:
                skipped += 1
                continue
            scores = json.loads(ts) if isinstance(ts, str) else ts
            options = list(scores.keys())
            vals = [float(scores[o]) for o in options]
            if len(options) < 2 or len(options) > len(LETTERS) or max(vals) <= 0:
                skipped += 1
                continue
            items.append({"uuid": r["uuid"], "config": cfg, "name": r["name"], "question": ex["input"],
                          "options": options, "scores": vals, "difficulty": difficulty_of(r["keywords"])})
    items.sort(key=lambda x: x["uuid"])           # fixed order independent of loader
    random.Random(seed).shuffle(items)            # fixed, seeded order so --limit is a reproducible subset
    if limit:
        items = items[:limit]
    return items, skipped


def bench_texts(items):
    return [it["question"] + " " + " ".join(it["options"]) for it in items]


HEADER = "The following is a multiple choice question about chemistry. Answer with the letter of the correct option.\n\n"


def build_prompt(item, passages=None, max_chars=900):
    s = HEADER
    if passages:
        s += ("Background passages retrieved from the ChemPile corpus (they may be irrelevant or wrong; "
              "they are retrieved text, not verified facts):\n")
        for i, p in enumerate(passages, 1):
            s += f"[{i}] {p['text'][:max_chars]}\n"
        s += "\n"
    s += f"Question: {item['question']}\n"
    for L, o in zip(LETTERS, item["options"]):
        s += f"{L}. {o}\n"
    s += "Answer:"
    return s


def letter_token_ids(tokenizer, n):
    """Token ids for each option letter, with and without leading space. Returns list of lists."""
    out = []
    for L in LETTERS[:n]:
        ids = set()
        for variant in (" " + L, L):
            t = tokenizer.encode(variant, add_special_tokens=False)
            if t:
                ids.add(t[-1])
        out.append(sorted(ids))
    return out


def pick_answer(logprobs_per_token, letter_ids):
    """logprobs_per_token: tensor/array vocab logprobs. Returns (best_index, per-option logsumexp list)."""
    vals = []
    for ids in letter_ids:
        xs = [float(logprobs_per_token[i]) for i in ids]
        m = max(xs)
        vals.append(m + math.log(sum(math.exp(x - m) for x in xs)))
    best = max(range(len(vals)), key=lambda i: vals[i])
    return best, vals


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - a) / d, (c + a) / d)
