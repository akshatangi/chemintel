"""Stream a ChemPile subset from Hugging Face, drop documents that overlap ChemBench, write jsonl.

Default: jablonkagroup/chempile-education, config LibreText_Chemistry-default (open chemistry textbooks).
The ChemPile-Education card states CC BY-NC-SA 4.0: non-commercial, attribution, share-alike. Outputs of this script
are adapted material under that licence (see DATA_LICENSE.md). Other ChemPile subsets have other licences.
Outputs: <out>/train.jsonl, <out>/valid.jsonl, <out>/prep_stats.json
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chemintel.bench import load_chembench, bench_texts
from chemintel.decontam import Decontaminator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="jablonkagroup/chempile-education")
    ap.add_argument("--config", default="LibreText_Chemistry-default")
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-train-docs", type=int, default=20000)
    ap.add_argument("--max-valid-docs", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=500)
    ap.add_argument("--max-chars-per-doc", type=int, default=20000)
    ap.add_argument("--ngram", type=int, default=10)
    a = ap.parse_args()
    from datasets import load_dataset
    os.makedirs(a.out, exist_ok=True)
    items, skipped = load_chembench()
    dec = Decontaminator(bench_texts(items), n=a.ngram)
    stats = {"dataset": a.dataset, "config": a.config, "chembench_mcq_questions": len(items), "chembench_non_mcq_skipped": skipped}
    for split, cap, name in (("train", a.max_train_docs, "train.jsonl"), ("val", a.max_valid_docs, "valid.jsonl")):
        ds = load_dataset(a.dataset, a.config, split=split, streaming=True)
        kept = chars = 0
        with open(os.path.join(a.out, name), "w") as f:
            for r in ds:
                t = r["text"]
                if len(t) < a.min_chars:
                    continue
                if dec.is_contaminated(t):
                    continue
                t = t[:a.max_chars_per_doc]
                f.write(json.dumps({"id": f"{split}-{kept}", "text": t, "source": ("LibreTexts: " + r["url"]) if r.get("url") else ""}) + "\n")
                kept += 1
                chars += len(t)
                if kept >= cap:
                    break
        stats[split] = {"docs": kept, "chars": chars, "approx_tokens": chars // 4}
    stats["decontamination"] = dec.stats()
    json.dump(stats, open(os.path.join(a.out, "prep_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
