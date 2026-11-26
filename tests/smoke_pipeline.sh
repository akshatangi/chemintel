#!/usr/bin/env bash
# CPU smoke test of the code path (training + evaluation with retrieval) using a TINY RANDOM model.
# It proves the scripts run end to end. The numbers it prints are meaningless (random weights).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 - <<'PY'
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, vocab_size=len(tok), n_positions=1024)
m = GPT2LMHeadModel(cfg); m.save_pretrained("/tmp/tiny_gpt2"); tok.save_pretrained("/tmp/tiny_gpt2")
print("tiny model saved")
PY
python3 scripts/03_train_qlora.py --model /tmp/tiny_gpt2 --data data/train.jsonl --out /tmp/smoke_adapter \
  --no-4bit --seq-len 128 --max-tokens 20000 --batch-size 2 --grad-accum 2 --max-steps 5 \
  --target-modules c_attn,c_proj --lora-r 4 --lora-alpha 8
python3 scripts/05_eval_chembench.py --backend hf --model /tmp/tiny_gpt2 --adapter /tmp/smoke_adapter --rag --index data/bm25.pkl \
  --limit 20 --name smoke_tiny_random_rag --out /tmp/smoke_results --max-prompt-tokens 512
python3 scripts/06_report.py /tmp/smoke_results
