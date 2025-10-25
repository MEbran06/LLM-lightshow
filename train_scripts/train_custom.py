# train.py — LED next-token fine-tune + export for llama2.c
# Minimal single-GPU/CPU trainer that:
#   1) loads your custom SentencePiece tokenizer (.model)
#   2) tokenizes your corpus (next-token prediction)
#   3) trains a small Transformer (from llama2.c/model.py)
#   4) exports weights to llama2.c-compatible model.bin

import os
import math
import time
import argparse
from contextlib import nullcontext
from pathlib import Path
import random

import torch
from sentencepiece import SentencePieceProcessor

# these come from the llama2.c repo
from model import Transformer, ModelArgs
from export import model_export


# ---------------------------
# Helpers
# ---------------------------
def pick_device(preferred: str = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    # mac
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_corpus_text(corpus_path: str) -> str:
    with open(corpus_path, "r", encoding="utf-8") as f:
        return f.read()


def build_token_stream(sp: SentencePieceProcessor, text: str, add_bos: bool = True, add_eos: bool = True):
    # encode the full text; SentencePiece handles whitespace
    # you can also split by lines and inject BOS/EOS between lines if preferred
    ids = []
    if add_bos:
        ids.append(sp.bos_id())
    ids.extend(sp.encode(text))
    if add_eos:
        ids.append(sp.eos_id())
    return ids


def split_ids(ids, train_ratio=0.9, seed=1337):
    random.Random(seed).shuffle(ids)  # light shuffle to de-correlate contiguous blocks a bit
    n = len(ids)
    n_train = int(n * train_ratio)
    return ids[:n_train], ids[n_train:]


def get_batch(id_tensor, batch_size, block_size, device):
    # sample random starting positions
    n = id_tensor.size(0) - block_size - 1
    ix = torch.randint(high=n, size=(batch_size,))
    x = torch.stack([id_tensor[i : i + block_size] for i in ix])
    y = torch.stack([id_tensor[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


def estimate_loss(model, ctx, train_ids, val_ids, batch_size, block_size, eval_iters, device):
    out = {}
    model.eval()
    with torch.no_grad():
        for split_name, split_ids in [("train", train_ids), ("val", val_ids)]:
            losses = []
            for _ in range(eval_iters):
                X, Y = get_batch(split_ids, batch_size, block_size, device)
                with ctx:
                    logits = model(X, Y)
                    loss = model.last_loss
                losses.append(loss.item())
            out[split_name] = sum(losses) / len(losses)
    model.train()
    return out


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Fine-tune tiny Transformer on LED corpus and export for llama2.c")
    ap.add_argument("--corpus", type=str, default="../dataset/train/corpus.txt",
                    help="Path to training text corpus (combined trained_*.txt)")
    ap.add_argument("--sp_model", type=str, default="../dataset/train/trained_corpus.model",
                    help="SentencePiece .model path (for training time tokenization)")
    ap.add_argument("--out_dir", type=str, default="../checkpoints/led_finetune",
                    help="Output directory for checkpoints and exported model.bin")
    ap.add_argument("--device", type=str, default=None, help="cpu|cuda|mps (auto if not set)")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=128, help="max sequence length (context)")
    ap.add_argument("--max_iters", type=int, default=2000)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_iters", type=int, default=100)
    ap.add_argument("--decay_lr", action="store_true", help="use cosine decay lr schedule")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--compile", action="store_true", help="torch.compile for speed (PyTorch 2.x)")
    # tiny model defaults (adjust if you want a bigger one)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv_heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--multiple_of", type=int, default=32)
    args = ap.parse_args()

    # device & dtype
    device = pick_device(args.device)
    device_type = "cuda" if device.startswith("cuda") else ("mps" if device == "mps" else "cpu")
    ptdtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.autocast(device_type, dtype=ptdtype)

    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.out_dir, exist_ok=True)

    # tokenizer
    if not os.path.exists(args.sp_model):
        raise FileNotFoundError(f"SentencePiece model not found: {args.sp_model}")
    sp = SentencePieceProcessor(model_file=args.sp_model)
    vocab_size = sp.vocab_size()

    # data
    if not os.path.exists(args.corpus):
        raise FileNotFoundError(f"Corpus not found: {args.corpus}")
    text = read_corpus_text(args.corpus)
    ids = build_token_stream(sp, text, add_bos=True, add_eos=True)
    # to torch tensor (single long stream)
    ids_t = torch.tensor(ids, dtype=torch.long)
    # split
    train_ids_list, val_ids_list = split_ids(ids, train_ratio=0.9, seed=args.seed)
    train_ids = torch.tensor(train_ids_list, dtype=torch.long)
    val_ids = torch.tensor(val_ids_list, dtype=torch.long)

    # model
    margs = ModelArgs(
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        n_kv_heads=args.kv_heads,
        vocab_size=vocab_size,
        multiple_of=args.multiple_of,
        max_seq_len=args.block_size,
        dropout=args.dropout,
    )
    model = Transformer(margs).to(device)

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model (PyTorch 2.x)...")
        model = torch.compile(model)

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=args.lr, betas=(0.9, 0.95), device_type=device_type)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16" and device_type == "cuda"))

    # lr schedule (optional cosine)
    def get_lr(it):
        if not args.decay_lr:
            return args.lr
        # warmup + cosine decay to 0
        if it < args.warmup_iters:
            return args.lr * it / max(1, args.warmup_iters)
        decay_ratio = (it - args.warmup_iters) / max(1, (args.max_iters - args.warmup_iters))
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * args.lr

    # training loop
    model.train()
    t0 = time.time()
    best_val = float("inf")
    for it in range(1, args.max_iters + 1):
        # set lr
        lr = get_lr(it)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # batch
        X, Y = get_batch(train_ids, args.batch_size, args.block_size, device)

        # fwd / bwd
        optimizer.zero_grad(set_to_none=True)
        if device_type == "cuda" and args.dtype == "float16":
            with ctx:
                logits = model(X, Y)
                loss = model.last_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with ctx:
                logits = model(X, Y)
                loss = model.last_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if it % 10 == 0:
            dt = time.time() - t0
            print(f"{it:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt*1000:.1f} ms/10 iters")
            t0 = time.time()

        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, ctx, train_ids, val_ids, args.batch_size, args.block_size, args.eval_iters, device)
            print(f"[eval] step {it}: train {losses['train']:.4f} | val {losses['val']:.4f}")
            # checkpoint best
            if losses["val"] < best_val:
                best_val = losses["val"]
                ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_args": margs.__dict__,
                        "iter": it,
                        "val_loss": best_val,
                    },
                    ckpt_path,
                )
                print(f"✓ saved checkpoint → {ckpt_path}")

    # final export to llama2.c format
    export_path = os.path.join(args.out_dir, "model.bin")
    # unwrap compiled module if needed
    raw_model = model
    try:
        raw_model = model._orig_mod  # if torch.compile wrapped it
    except Exception:
        pass
    model_export(raw_model, export_path, version=0)
    print(f"✅ exported llama2.c checkpoint → {export_path}")

    print("\n=== How to run in C (llama2.c) ===")
    print("1) Make sure you also export your tokenizer to .bin (using your tokenizer.py).")
    print("2) Example run:\n")
    print(f"   cd llama2.c")
    print(f"   make run")
    print(f'   ./run {os.path.relpath(export_path, start="llama2.c")} -z ../dataset/train/trained_corpus.bin -i "F 10 L0.20,0.15,0.09,0.02,0.01,0.009,0.010,0.006 ->" -n 64\n')


if __name__ == "__main__":
    main()