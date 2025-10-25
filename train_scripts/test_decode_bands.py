#!/usr/bin/env python3
import argparse, random, re, sys
from pathlib import Path

LINE_RE = re.compile(r"^F\s+(\d+)\s+(NULL|L([0-9.,]+))")

def parse_line(line: str):
    """
    Returns (frame_idx:int, is_null:bool, bands:list[float] or None)
    Expected formats:
      F 123 NULL -> ...
      F 123 L0.204,0.261,0.086,0.021,0.011,0.009,0.010,0.006 -> ...
    """
    m = LINE_RE.match(line.strip())
    if not m:
        return None
    t = int(m.group(1))
    if m.group(2) == "NULL":
        return (t, True, None)
    bands_str = m.group(3)
    bands = [float(x) for x in bands_str.split(",")]
    return (t, False, bands)

def main():
    ap = argparse.ArgumentParser(description="Sample and decode 8-band frames from corpus/trained txt.")
    ap.add_argument("--dataset", required=True, help="Path to corpus or trained_XXX.txt")
    ap.add_argument("--n", type=int, default=5, help="How many random lines to sample")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    ap.add_argument("--plot", action="store_true", help="If set, try to plot sampled bands")
    args = ap.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with path.open("r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip().startswith("F ")]

    if not lines:
        print("No frame lines found (lines starting with 'F ').", file=sys.stderr)
        sys.exit(1)

    random.seed(args.seed)
    sample = random.sample(lines, k=min(args.n, len(lines)))

    print(f"\nFile: {path}  |  Total frame lines: {len(lines)}  |  Showing {len(sample)} random frames\n")
    parsed = []
    for ln in sample:
        p = parse_line(ln)
        if not p:
            continue
        t, is_null, bands = p
        if is_null:
            print(f"F {t:>6}  ->  NULL (all LEDs OFF)")
        else:
            # bands is length 8
            bands_fmt = ", ".join(f"{b:.3f}" for b in bands)
            print(f"F {t:>6}  ->  Bands[8] = [{bands_fmt}]")
        parsed.append(p)

    # Optional plotting
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception as e:
            print(f"\n[plot] Skipping plot; matplotlib not available ({e}).")
            return

        non_null = [(t,b) for (t,is_null,b) in parsed if not is_null]
        if not non_null:
            print("\n[plot] No non-NULL frames in sample to plot.")
            return

        cols = 1
        rows = len(non_null)
        fig, axes = plt.subplots(rows, cols, figsize=(6, 2.0*rows), squeeze=False)
        for ax, (t, bands) in zip(axes[:,0], non_null):
            ax.bar(np.arange(len(bands)), bands)
            ax.set_title(f"Frame F={t} (8 bands)")
            ax.set_xlabel("Band index (0..7)")
            ax.set_ylabel("Amplitude (norm)")
            ax.set_ylim(0, max(0.001, max(bands)*1.1))
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()