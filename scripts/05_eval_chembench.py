"""Evaluate on ChemBench multiple-choice questions.

Backends
  hf              local model, scored by next-token log-probability of the option letters (no generation parsing).
  retrieval_only  NO language model: retrieve passages for the question and pick the option whose words appear most
                  in them (BM25-style overlap). A model-free retrieval baseline, expected to be weak.
  chance          picks uniformly at random (seeded). Reference line.
Conditions: --rag adds top-k retrieved ChemPile passages to the prompt (hf) or is required (retrieval_only).
Output: JSON with overall / per-subfield / per-difficulty accuracy (95% Wilson interval) and every prediction.
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chemintel.bench import load_chembench, build_prompt, letter_token_ids, pick_answer, wilson
from chemintel.bm25 import BM25Index, tokenize


def retrieval_only_pick(item, passages):
    ctx = set()
    for p in passages:
        ctx.update(tokenize(p["text"]))
    scores = []
    for o in item["options"]:
        toks = set(tokenize(o))
        scores.append(len(toks & ctx) / max(1, len(toks)))
    return max(range(len(scores)), key=lambda i: scores[i]), scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "retrieval_only", "chance"], default="hf")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--rag", action="store_true")
    ap.add_argument("--index", default="data/bm25.pkl")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="0 = all MCQ questions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subfields", default="")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-prompt-tokens", type=int, default=2048)
    a = ap.parse_args()
    if a.backend == "retrieval_only":
        a.rag = True
    items, skipped = load_chembench(subfields=a.subfields.split(",") if a.subfields else None, seed=a.seed, limit=a.limit or None)
    index = BM25Index.load(a.index) if a.rag else None
    if a.backend == "hf":
        import torch
        from chemintel.modeling import load_model, device_of
        tok, model = load_model(a.model, adapter=a.adapter)
        dev = device_of(model)
        maxlen = a.max_prompt_tokens
    rng = random.Random(a.seed)
    preds, t0 = [], time.time()
    for n, it in enumerate(items):
        passages = index.search(it["question"], k=a.k) if index else None
        if a.backend == "chance":
            best, vals = rng.randrange(len(it["options"])), None
        elif a.backend == "retrieval_only":
            best, vals = retrieval_only_pick(it, passages)
        else:
            prompt = build_prompt(it, passages)
            ids = tok(prompt, return_tensors="pt").input_ids[:, -maxlen:].to(dev)   # keep the question end if truncated
            with torch.no_grad():
                logits = model(input_ids=ids).logits[0, -1].float()
            lp = torch.log_softmax(logits, -1).cpu()
            best, vals = pick_answer(lp, letter_token_ids(tok, len(it["options"])))
        preds.append({"uuid": it["uuid"], "config": it["config"], "difficulty": it["difficulty"], "pred": best,
                      "correct": it["scores"][best] > 0, "n_options": len(it["options"]),
                      "sources": [p["source"] for p in passages] if passages else None})
        if (n + 1) % 50 == 0:
            print(f"{n+1}/{len(items)} acc so far {sum(p['correct'] for p in preds)/len(preds):.3f}", flush=True)

    def agg(sel):
        k = sum(p["correct"] for p in sel); n = len(sel); lo, hi = wilson(k, n)
        return {"n": n, "correct": k, "accuracy": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)]}
    res = {"name": a.name or f"{a.backend}{'+rag' if a.rag else ''}", "backend": a.backend, "model": a.model if a.backend == "hf" else None,
           "adapter": a.adapter, "rag": a.rag, "k": a.k if a.rag else None, "index": a.index if a.rag else None,
           "seed": a.seed, "limit": a.limit, "mcq_questions": len(items), "non_mcq_skipped": skipped,
           "chance_expected": round(sum(1 / p["n_options"] for p in preds) / len(preds), 4),
           "overall": agg(preds),
           "by_subfield": {c: agg([p for p in preds if p["config"] == c]) for c in sorted({p["config"] for p in preds})},
           "by_difficulty": {d: agg([p for p in preds if p["difficulty"] == d]) for d in sorted({p["difficulty"] for p in preds})},
           "seconds": round(time.time() - t0, 1), "predictions": preds}
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, res["name"].replace("/", "_") + ".json")
    json.dump(res, open(path, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("name", "mcq_questions", "chance_expected", "overall", "by_difficulty", "seconds")}, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
