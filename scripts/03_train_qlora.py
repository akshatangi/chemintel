"""QLoRA training on a text jsonl.

--mode cpt : continued pretraining. Concatenate documents (with EOS), cut into fixed-length blocks, train on all tokens.
--mode sft : jsonl rows {"prompt","completion"}; loss only on the completion.
The base weights stay frozen (4-bit on GPU); only LoRA adapters train. Writes the adapter + train_log.jsonl.
Plain PyTorch loop on purpose (fewer library-version surprises than Trainer/TRL).
"""
import argparse, json, math, os, random, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def build_cpt_blocks(path, tok, seq_len, max_tokens, seed):
    ids, total = [], 0
    for line in open(path):
        t = json.loads(line)["text"]
        x = tok.encode(t, add_special_tokens=False) + [tok.eos_token_id]
        ids.extend(x)
        total += len(x)
        if max_tokens and total >= max_tokens:
            break
    blocks = [ids[i:i + seq_len] for i in range(0, len(ids) - seq_len + 1, seq_len)]
    random.Random(seed).shuffle(blocks)
    return [(b, b) for b in blocks], total


def build_sft_examples(path, tok, seq_len, max_tokens, seed):
    ex, total = [], 0
    for line in open(path):
        r = json.loads(line)
        p = tok.encode(r["prompt"], add_special_tokens=False)
        c = tok.encode(r["completion"], add_special_tokens=False) + [tok.eos_token_id]
        x = (p + c)[:seq_len]
        lab = ([-100] * len(p) + c)[:seq_len]
        ex.append((x, lab))
        total += len(x)
        if max_tokens and total >= max_tokens:
            break
    random.Random(seed).shuffle(ex)
    return ex, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--out", default="runs/cpt")
    ap.add_argument("--mode", choices=["cpt", "sft"], default="cpt")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=5_000_000, help="token budget; 0 = all data")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke tests / time box)")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--no-4bit", action="store_true", help="disable 4-bit (CPU smoke tests)")
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from chemintel.modeling import load_model, device_of
    random.seed(a.seed); torch.manual_seed(a.seed)
    four = False if a.no_4bit else None
    tok, model = load_model(a.model, four_bit=four, trainable=True)
    is4 = bool(getattr(model, "is_loaded_in_4bit", False)) or getattr(model.config, "quantization_config", None) is not None
    if is4:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=a.grad_checkpointing)
    elif a.grad_checkpointing:
        model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
                     target_modules=a.target_modules.split(","), task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    dev = device_of(model)

    builder = build_cpt_blocks if a.mode == "cpt" else build_sft_examples
    examples, ntok = builder(a.data, tok, a.seq_len, a.max_tokens, a.seed)
    assert examples, "no training examples built; check --data / --seq-len"
    steps_per_epoch = max(1, len(examples) // (a.batch_size * a.grad_accum))
    total_steps = max(1, int(steps_per_epoch * a.epochs))
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    print(f"examples={len(examples)} tokens_read={ntok} optimizer_steps={total_steps}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warm = max(1, int(0.03 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    use_amp = dev.type == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "train_log.jsonl"), "w")
    json.dump({**vars(a), "examples": len(examples), "tokens_read": ntok, "optimizer_steps": total_steps},
              open(os.path.join(a.out, "train_args.json"), "w"), indent=1)

    def batchify(chunk):
        L = max(len(x) for x, _ in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id); lab = torch.full((len(chunk), L), -100)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for i, (x, y) in enumerate(chunk):
            ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
        return ids.to(dev), lab.to(dev), att.to(dev)

    model.train()
    t0, pos = time.time(), 0
    for step in range(total_steps):
        loss_sum = 0.0
        for _ in range(a.grad_accum):
            if pos + a.batch_size > len(examples):
                random.shuffle(examples); pos = 0
            ids, lab, att = batchify(examples[pos:pos + a.batch_size]); pos += a.batch_size
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_amp):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / a.grad_accum
            scaler.scale(loss).backward()
            loss_sum += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        rec = {"step": step + 1, "loss": round(loss_sum, 4), "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)}
        logf.write(json.dumps(rec) + "\n"); logf.flush()
        if (step + 1) % 10 == 0 or step == 0:
            print(rec)
        if (step + 1) % a.save_every == 0:
            model.save_pretrained(a.out)
    model.save_pretrained(a.out)
    print("saved adapter to", a.out)


if __name__ == "__main__":
    main()
