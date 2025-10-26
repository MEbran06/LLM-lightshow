import argparse
import numpy as np
from tokenizer import Tokenizer
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="tokenizer.model or tokenizer.bin")
    ap.add_argument("--input_txt", required=True, help="training pairs text file")
    ap.add_argument("--out_dir", default="dataset", help="output directory")
    ap.add_argument("--val_frac", type=float, default=0.1, help="fraction used for validation")
    args = ap.parse_args()

    enc = Tokenizer(args.tokenizer)
    lines = [l.strip() for l in open(args.input_txt, "r", encoding="utf-8") if l.strip()]

    ids = []
    for line in lines:
        token_ids = enc.encode(line, bos=False, eos=False)
        ids.extend(token_ids)

    ids = np.array(ids, dtype=np.uint16)
    out_path = Path(args.out_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    split = int(len(ids) * (1 - args.val_frac))
    ids[:split].tofile(out_path / "train.bin")
    ids[split:].tofile(out_path / "val.bin")

    print(f" Saved train.bin ({split} tokens)")
    print(f" Saved val.bin ({len(ids)-split} tokens)")

    # Validate minibatch sample decoding
    test_ids = ids[:32]
    print("Sample decode:", enc.decode([int(x) for x in test_ids.tolist()]))


if __name__ == "__main__":
    main()
