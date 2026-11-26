# Data licences and attribution

This project downloads data at run time. The licences below apply to that data and to anything derived from it.
Sources and licence statements were read on 26 Nov 2026 from the pages linked; re-check them before you reuse anything.

| Data | Use here | Licence (as stated by the source) | Link |
|---|---|---|---|
| ChemPile-Education (LibreText Chemistry config) | Retrieval corpus and optional continued-pretraining text | CC BY-NC-SA 4.0: non-commercial, attribution, share-alike | https://huggingface.co/datasets/jablonkagroup/chempile-education |
| ChemBench | Evaluation questions only (never used for training or retrieval) | MIT (Hugging Face dataset metadata); code repo MIT | https://huggingface.co/datasets/jablonkagroup/ChemBench , https://github.com/lamalab-org/chembench |

Derived artifacts: `data/train.jsonl`, `data/valid.jsonl` and `data/bm25.pkl` are filtered / chunked / indexed copies of
ChemPile-Education text. Changes made: documents shorter than 500 characters removed, documents with a 10-word overlap with any ChemBench
question or the ChemBench canary removed, long documents truncated to 20,000 characters, text split into ~900 character passages.
Any redistribution of these files must keep this notice, credit the ChemPile authors, stay non-commercial and carry CC BY-NC-SA 4.0.
Any model adapter trained on that text is likely a derivative as well; check with your instructor before publishing it.

Please cite: Mirza et al., "ChemPile: A 250GB Diverse and Curated Dataset for Chemical Foundation Models" (arXiv:2505.12534, https://arxiv.org/abs/2505.12534)
and Mirza et al., "Are large language models superhuman chemists?" (arXiv:2404.01475) for ChemBench.
