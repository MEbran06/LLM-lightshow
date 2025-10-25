# combine_led_dataset.py
# ✅ Automatically combines all *_features.txt and *_commands.txt files in dataset/test/
# ✅ Preserves frame IDs (like F 0, F 1...)
# ✅ Handles NULL lines properly (forces all OFF)
# ✅ Saves to dataset/train/trained_<song>.txt for each song

from pathlib import Path

# === Paths ===
test_dir = Path("dataset/test")
train_dir = Path("dataset/train")
train_dir.mkdir(parents=True, exist_ok=True)

# automatically detect all song IDs (e.g., 001, 002, 010, etc.)
songs = sorted({p.stem.split("_")[0] for p in test_dir.glob("*_features.txt")})
print(f"🎵 Found {len(songs)} songs: {songs}")

# === Process Each Song ===
for sid in songs:
    feat_path = test_dir / f"{sid}_features.txt"
    cmd_path = test_dir / f"{sid}_commands.txt"
    out_path = train_dir / f"trained_{sid}.txt"

    if not feat_path.exists() or not cmd_path.exists():
        print(f"⚠️ Skipping {sid} — missing files.")
        continue

    with open(feat_path) as f_feat, open(cmd_path) as f_cmd, open(out_path, "w") as f_out:
        for feat_line, cmd_line in zip(f_feat, f_cmd):
            feat_line = feat_line.strip()
            cmd_line = cmd_line.strip()

            # preserve frame prefix (e.g., F 10)
            parts = feat_line.split(" ", 1)
            frame_prefix = parts[0] if len(parts) > 1 else "F 0"
            rest = parts[1] if len(parts) > 1 else ""

            # handle NULL lines — all LEDs OFF
            if "NULL" in feat_line:
                cmd_line = " ".join([f"B{i}:OFF" for i in range(8)])

            # write combined formatted line
            f_out.write(f"{frame_prefix} {rest} -> {cmd_line}\n")

    print(f"✅ Combined {sid} → {out_path}")

print("🎯 All LED datasets combined successfully!")