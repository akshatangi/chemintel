# ChemIntel: retrieval-grounded chemistry Q&A (ChemPile + ChemBench)

A runnable package built on 26 Nov 2026. It retrieves passages from a ChemPile-Education (LibreTexts) subset, answers with a language
model you connect, and evaluates on ChemBench multiple-choice questions with and without retrieval.

## Status, stated plainly
| Part | Status |
|---|---|
| Corpus download, ChemBench overlap filter, BM25 index (`scripts/01`, `02`) | **Run and checked** on CPU (3,000 documents, 28,820 passages) |
| Retrieval demo, no generator (`app.py`) | **Run**; screenshot in `docs/demo_retrieval_only.png` |
| ChemBench evaluation harness (`scripts/05`) | **Run** for two model-free baselines on 2,542 questions; run for `hf` backend only with a tiny random model |
| Training script (`scripts/03`, QLoRA) | **Smoke-tested** on CPU with a tiny random model for 5 steps (code path works). **Never run on a real model or GPU. Optional, not needed for the RAG version.** |
| SmolLM2-**135M**-Instruct (same family, not the 1.7B default) smoke test, chat template + LoRA target-module names | **Run** (`scripts/04`): loads and answers; answer quality is poor, nothing measured |
| Ollama / OpenAI-compatible generator client | **Tested against a mock server only** (`tests/test_openai_client_mock.py`). Not tested with any real model |
| Language-model accuracy on ChemBench (base, +retrieval, +training) | **Not measured. No numbers exist.** |
| Datasheet field extraction (`chemintel/datasheet.py`) | **Regex method run on 3 real public data sheets and compared by hand** (0 wrong values, but misses some; see `docs/REAL_DATA_EVAL.md`), plus unit tests. Heuristic, not validated by a chemist. LLM method: code and fake-model test only, never run with a real model |
| Reaction precedent retrieval (`chemintel/reactions.py`, importer `scripts/07_import_ord.py`) | **Run on 750 real ORD reactions** (one narrow N-arylation subset) with hand-read query checks, plus unit tests on synthetic records. Word-based retrieval, no structure search, no feasibility or yield prediction. ORD data is not shipped (CC BY-SA 4.0). See `docs/REAL_DATA_EVAL.md` |

## Measured so far (no language model involved)
ChemBench MCQ, 2,542 questions (243 numeric questions skipped), fixed seed 0, files in `results/`:

| run | accuracy | 95% interval (Wilson) |
|---|---|---|
| random choice (seeded) | 0.359 (expected chance given option counts: 0.333) | 0.340 - 0.378 |
| BM25 retrieval-only heuristic (pick the option whose words appear most in 3 retrieved passages) | 0.374 | 0.355 - 0.393 |

Reading: the retrieval-only heuristic is not convincingly better than guessing. It is a floor to beat, not a result for your system.
**Difficulty split is thin:** only 19 questions carry a `difficulty-advanced` tag, 689 are `basic`, 1,834 carry neither. Do not draw
basic-vs-advanced conclusions from this dataset's tags without checking the counts printed in each result file.

## Quick start (verified from a clean copy of this folder, CPU sandbox, Linux; not tested on macOS or Windows)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # includes torch (large). bitsandbytes is for GPU training only
python scripts/01_prepare_corpus.py --out data --max-train-docs 3000   # streams ChemPile-Education, filters ChemBench overlap
python scripts/02_build_index.py                                        # BM25 index (about 340 MB RAM for 3,000 docs)
python scripts/04_smoke_generator.py --backend hf --model HuggingFaceTB/SmolLM2-1.7B-Instruct   # ONE question: does the model load and answer?
streamlit run app.py                                                    # UI: ChemPile and/or an uploaded text PDF
```
Step 04 downloads about 3.4 GB of weights (1.7B parameters; size is my estimate, not from the model card) and needs roughly 7 GB free RAM at fp32 (the default load). On a small Mac use the Ollama route below, or `--model HuggingFaceTB/SmolLM2-360M-Instruct`.
Only after step 04 answers sensibly, run the bounded benchmarks below.

### Connect a generator (free, local): Ollama
1. Install Ollama (macOS 14 Sonoma or newer; Apple M-series supported per https://docs.ollama.com/macos) from https://ollama.com/download.
2. `ollama pull qwen2.5:1.5b` (listed at 986 MB; Apache 2.0 per https://ollama.com/library/qwen2.5).
3. In the app sidebar choose *OpenAI-compatible endpoint*, base URL `http://localhost:11434/v1`, model `qwen2.5:1.5b`, any/blank key.
A 1.5B general model is weak at chemistry, especially SMILES and calculations. Its accuracy here is unmeasured.

### Evaluate with and without retrieval
```bash
python scripts/05_eval_chembench.py --backend hf --model HuggingFaceTB/SmolLM2-1.7B-Instruct --limit 100 --name base_noRAG
python scripts/05_eval_chembench.py --backend hf --model HuggingFaceTB/SmolLM2-1.7B-Instruct --limit 100 --rag --name base_RAG
python scripts/06_report.py results
```
`--limit` takes a fixed seeded subset, so both runs see the same questions. Start with `--limit 100` (a bounded, paired run); raise it only if time allows. Use a GPU runtime (Colab notebook `ChemIntel_Colab.ipynb`); on CPU a 1.5B model is slow.
The `hf` backend scores the option letters by log-probability, so no answer parsing is involved. Retrieved passages are placed in the prompt
and labelled as possibly irrelevant retrieved text.

### Optional training (not run, not needed for RAG)
`scripts/03_train_qlora.py` does continued pretraining (`--mode cpt`) or instruction tuning (`--mode sft`) with 4-bit QLoRA
(Dettmers et al., https://arxiv.org/abs/2305.14314). Start with `--max-steps 50` to see speed and memory. This is not distillation:
distillation (Hinton et al., https://arxiv.org/abs/1503.02531) would need teacher-model outputs as training targets, which is not implemented.
A few million tokens of ChemPile text is a small amount; it may not change ChemBench accuracy. Report what you measure.

## Leakage prevention
- ChemBench is used for evaluation only. It carries a canary string asking that it never appear in training data.
- `chemintel/decontam.py` drops any corpus document sharing a 10-word sequence with a ChemBench question or option text, or containing the canary
  (12 of 3,112 documents dropped in our run). It is a heuristic: questions under 10 words are not covered, paraphrases are not caught,
  and it does not prove the corpus is clean.
- Retrieved passages come from the filtered corpus. Evaluation questions are never added to the index.

## Datasheet extraction and reaction precedents (implemented minimally; see status table)
- `chemintel/datasheet.py`: regex extraction of density, flash/boiling/melting point, auto-ignition, pot life, shelf life, pH, H-statements and CAS numbers (checksum-validated) from an uploaded text PDF, with page numbers and the evidence text. This is **heuristic**. An optional LLM method keeps a model's value only if it appears literally in the cited passage; that proves presence in the text, not correctness. The LLM method has never been run with a real model. No scoring against hand-labelled datasheets has been done.
- `chemintel/reactions.py`: retrieves similar recorded reactions from a JSON-lines file; query words must mostly appear in a record or nothing is returned. `scripts/07_import_ord.py` builds that file from ORD parquet files. Only 2 clearly SYNTHETIC records ship in `examples/`. It is precedent retrieval, not a validated feasibility prediction. ORD (https://github.com/open-reaction-database/ord-data, CC BY-SA 4.0) was loaded for one subset only.

## Data and licences
See `DATA_LICENSE.md`. This package does **not** include the corpus or the index: download them with the scripts. ChemPile-Education is
CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike) per its dataset card; we did not verify the licences of every underlying LibreTexts page,
so do not redistribute the data or index outside coursework without checking.
Cite ChemPile (arXiv:2505.12534) and ChemBench (arXiv:2404.01475).

## Tests
`python tests/test_units.py` (BM25, chunking, overlap filter, prompt, scoring, PDF pages, datasheet regex and LLM-check, reaction precedents), `python tests/test_openai_client_mock.py`,
`bash tests/smoke_pipeline.sh` (tiny random model, needs network once for the gpt2 tokenizer and ChemBench).


## Minimal on-device smoke test (Mac, not yet run by us)
No language-model accuracy is claimed anywhere in this package; the only numbers are the two model-free baselines above.
1. `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (torch is large; skip bitsandbytes on a Mac).
2. `python scripts/01_prepare_corpus.py --out data --max-train-docs 300 --max-valid-docs 20` then `python scripts/02_build_index.py` (small and quick).
3. `python scripts/04_smoke_generator.py --backend hf --model HuggingFaceTB/SmolLM2-1.7B-Instruct` ONE question. Weights are about 3.4 GB (our estimate) and need roughly 7 GB free RAM at fp32. If it runs out of memory or is slow, retry with `--model HuggingFaceTB/SmolLM2-360M-Instruct`.
4. `streamlit run app.py`, choose "Local Hugging Face model", upload a text PDF, ask a question.
Send back the printed output or the error text, the Mac model, and its RAM. Only after step 3 works, run `scripts/05_eval_chembench.py` with `--limit 100` (with and without `--rag`).

## Status (honest)
- Verified in a CPU-only sandbox: corpus prep + decontamination, BM25 index, model-free baselines (chance 0.359, BM25 retrieval-only heuristic 0.374 on 2,542 ChemBench MCQs; not LLM results), unit tests incl. PDF page indexing, mock OpenAI-client test, tiny-random-model smoke run of the train/eval code, Streamlit UI in retrieval-only mode (docs/*.png).
- NOT verified: SmolLM2-1.7B-Instruct running (default model; the chat template, MPS path and LoRA target modules are untested with real weights), any LLM accuracy, real QLoRA training. Run these on a Mac/Colab and report the numbers; do not quote any until measured.
- PDF upload tab: text-based PDFs only (pypdf, BM25 over pages). Datasheet extraction and reaction precedents exist as described in the table; only the small checks in `docs/REAL_DATA_EVAL.md` exist (3 data sheets, 750 reactions from one source).
- ChemBench "advanced" tag has only 19 questions, so basic-vs-advanced is not reliable. Decontamination (10-gram + canary) reduces, not eliminates, leakage.
- Corpus/index are not shipped (ChemPile-Education is CC BY-NC-SA 4.0); regenerate with scripts 01 and 02.

## Model resources and licence caveats
- Default model `HuggingFaceTB/SmolLM2-1.7B-Instruct`: Apache-2.0 per its Hugging Face metadata (https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct). Not run here at 1.7B (my sandbox has about 1 GB RAM).
- Never tested end to end: SmolLM2-1.7B generating in the UI, MPS on a Mac, evaluation with a real 1.7B model, real QLoRA. The package is not "end-to-end tested".
- Python dependencies carry their own licences; check before redistributing.
