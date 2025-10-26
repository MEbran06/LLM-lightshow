import struct
import os

def combine_corpus(prefix="trained_001"):
    model_file = f"{prefix}.model"
    vocab_file = f"{prefix}.vocab"
    bin_file   = f"{prefix}.bin"
    out_file   = f"{prefix}_corpus.bin"

    if not all(os.path.exists(f) for f in [model_file, vocab_file, bin_file]):
        print("❌ Missing one or more required files (.model, .vocab, .bin)")
        return

    with open(out_file, "wb") as out:
        for name, file in [("MODEL", model_file), ("VOCAB", vocab_file), ("BIN", bin_file)]:
            data = open(file, "rb").read()
            header = f"{name}:{len(data)}\n".encode("utf-8")
            out.write(header)
            out.write(data)
            out.write(b"\n===END===\n")

    print(f"✅ Combined into {out_file}")

if __name__ == "__main__":
    combine_corpus("trained_001")