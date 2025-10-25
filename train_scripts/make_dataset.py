#!/usr/bin/env python3
# ===============================================
# Hackathon 2025 — Feature + Command Dataset Builder (Sum-Energy Version)
# ===============================================

import argparse, os
from pathlib import Path
import numpy as np
import librosa

# ---------------- CONFIG ----------------
SR = 16000            # sample rate
N_FFT = 1024          # FFT size
HOP = 512             # hop size
NUM_BANDS = 8         # LED frequency bands
ENERGY_THRESH = 0.05  # frame energy threshold
SENSITIVITY = 1.3     # higher = fewer LEDs ON

# ---------------- BAND COMPUTATION ----------------
def compute_bands(y):
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    edges = np.geomspace(20, 16000, num=NUM_BANDS + 1)
    bands = []
    for i in range(NUM_BANDS):
        idx = np.where((freqs >= edges[i]) & (freqs < edges[i + 1]))[0]
        bands.append(np.mean(S[idx, :], axis=0) if len(idx) else np.zeros(S.shape[1]))
    B = np.vstack(bands)
    B /= np.max(B) + 1e-9
    return B

# ---------------- FRAME COMPRESSION ----------------
def compress_frames(bands, avg_n=1):
    if avg_n <= 1:
        return bands
    n_frames = bands.shape[1] // avg_n
    bands_c = np.zeros((bands.shape[0], n_frames))
    for i in range(n_frames):
        start = i * avg_n
        end = start + avg_n
        bands_c[:, i] = np.mean(bands[:, start:end], axis=1)
    return bands_c

# ---------------- COMMAND BUILDER ----------------
def build_grouped_commands_synced(bands, energy, energy_thresh, sensitivity):
    """
    For each frame:
      - if energy <= energy_thresh → ALL OFF
      - else threshold per frame = mean(frame) * sensitivity
    Returns one T<i> line per frame (same length as features).
    """
    out = []
    for i in range(bands.shape[1]):
        if energy[i] <= energy_thresh:
            leds = [f"B{b}:OFF" for b in range(bands.shape[0])]
            out.append(f"T{i} " + " ".join(leds))
            continue

        frame = bands[:, i]
        mean_energy = float(np.mean(frame))
        dynamic_thresh = mean_energy * sensitivity

        leds = []
        for b in range(bands.shape[0]):
            state = "ON" if frame[b] >= dynamic_thresh else "OFF"
            leds.append(f"B{b}:{state}")
        out.append(f"T{i} " + " ".join(leds))
    return out

# ---------------- PROCESS ONE SONG ----------------
def process_one(song_path: Path, out_dir: Path, avg_n: int, sensitivity: float):
    """Extract features + generate commands for one song (sum-based energy)."""
    print(f"🎧 Processing {song_path.name}")
    y, _ = librosa.load(str(song_path), sr=SR, mono=True)
    if not np.allclose(y, 0.0):
        y = librosa.util.normalize(y)

    # compute 8 frequency bands
    bands = compute_bands(y)
    bands_c = compress_frames(bands, avg_n=avg_n)

    # 🔥 NEW: use SUM instead of MEAN for total amplitude energy
    energy = np.sum(bands_c, axis=0)
    energy /= np.max(energy) + 1e-9  # normalize to 0–1

    # FEATURES
    feature_lines = []
    for i in range(bands_c.shape[1]):
        if energy[i] > ENERGY_THRESH:
            vals = ",".join(f"{bands_c[b, i]:.3f}" for b in range(bands_c.shape[0]))
            feature_lines.append(f"F {i} L{vals}")
        else:
            feature_lines.append(f"F {i} NULL")

    # COMMANDS — perfectly synced with features
    command_lines = build_grouped_commands_synced(
        bands=bands_c,
        energy=energy,
        energy_thresh=ENERGY_THRESH,
        sensitivity=sensitivity
    )

    # save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    base = song_path.stem
    (out_dir / f"{base}_features.txt").write_text("\n".join(feature_lines) + "\n")
    (out_dir / f"{base}_commands.txt").write_text("\n".join(command_lines) + "\n")

    print(f"✅ Done → {base}_features.txt / {base}_commands.txt")
    print(f"   Frames: {len(feature_lines)}  (commands match)\n")

# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--songs_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--avg_n", type=int, default=1)
    parser.add_argument("--sensitivity", type=float, default=SENSITIVITY)
    args = parser.parse_args()

    songs = sorted(Path(args.songs_dir).glob("*.wav"))
    if not songs:
        print("❌ No .wav files found in songs/")
        return

    for song in songs:
        process_one(song, Path(args.out_dir), args.avg_n, args.sensitivity)

if __name__ == "__main__":
    main()
