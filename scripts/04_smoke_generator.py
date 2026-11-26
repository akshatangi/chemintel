"""ONE-question generator smoke test. Run this first on your own machine, before any benchmark.
  python scripts/04_smoke_generator.py --backend hf --model HuggingFaceTB/SmolLM2-1.7B-Instruct
  python scripts/04_smoke_generator.py --backend openai --model qwen2.5:1.5b --base-url http://localhost:11434/v1
It prints the retrieved passages and the model's answer, and the time taken. It proves the model loads and answers; it measures nothing."""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from chemintel.bm25 import BM25Index
from chemintel.generators import HFGenerator, OpenAICompatGenerator, build_rag_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["hf", "openai"], default="hf")
ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
ap.add_argument("--base-url", default="http://localhost:11434/v1")
ap.add_argument("--index", default="data/bm25.pkl")
ap.add_argument("--question", default="Why does adding salt raise the boiling point of water?")
ap.add_argument("--max-new-tokens", type=int, default=120)
a = ap.parse_args()
t0 = time.time()
gen = HFGenerator(a.model) if a.backend == "hf" else OpenAICompatGenerator(a.model, a.base_url)
print(f"model ready in {time.time()-t0:.1f}s")
passages = BM25Index.load(a.index).search(a.question, k=3) if os.path.exists(a.index) else []
print(f"{len(passages)} passages retrieved")
t0 = time.time()
out = gen.generate(build_rag_prompt(a.question, passages) if passages else a.question, max_new_tokens=a.max_new_tokens)
print(f"--- answer ({time.time()-t0:.1f}s) ---\n{out}")
