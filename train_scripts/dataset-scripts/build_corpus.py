import argparse
import os
import glob
import random

def main():
    parser = argparse.ArgumentParser(description="Combine multiple .txt files into one corpus.")
    parser.add_argument("--input_dir", required=True, help="Directory containing train text files")
    parser.add_argument("--output_file", default="corpus.txt", help="Output file path")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle lines in corpus")
    args = parser.parse_args()

    print(f"Scanning {args.input_dir} for *.txt files")

    txt_files = glob.glob(os.path.join(args.input_dir, "**/*.txt"), recursive=True)
    assert len(txt_files) > 0, "No .txt files found in input_dir!"

    lines = []
    for file in txt_files:
        with open(file, "r", encoding="utf-8") as f:
            file_lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            lines.extend(file_lines)
            print(f"Read {len(file_lines)} lines from {file}")

    if args.shuffle:
        random.shuffle(lines)
        print("Shuffled lines")

    print(f"Total cleaned lines: {len(lines)}")

    with open(args.output_file, "w", encoding="utf-8") as out:
        for ln in lines:
            out.write(ln + "\n")

    print(f"✅ Corpus saved: {args.output_file}")
    print("Done!")

if __name__ == "__main__":
    main()
