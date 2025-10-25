# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import struct
import argparse
from typing import List
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

TOKENIZER_MODEL = "tokenizer.model"  # default llama sentencepiece tokenizer model

class Tokenizer:
    def __init__(self, tokenizer_model=None):
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def export(self):
        """Export tokenizer model to .bin for llama2.c"""
        tokens, scores = [], []
        for i in range(self.n_words):
            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            if i == self.bos_id:
                t = '\n<s>\n'
            elif i == self.eos_id:
                t = '\n</s>\n'
            t = t.replace('▁', ' ')
            b = t.encode('utf-8')
            tokens.append(b)
            scores.append(s)

        max_token_length = max(len(t) for t in tokens)
        tokenizer_bin = self.model_path.replace('.model', '.bin')

        with open(tokenizer_bin, 'wb') as f:
            f.write(struct.pack("I", max_token_length))
            for bytes_, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(bytes_)))
                f.write(bytes_)

        print(f"✅ Exported tokenizer to {tokenizer_bin}")

if __name__ == "__main__":
    # Use path relative to script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, "../dataset/train")
    corpus_path = os.path.join(train_dir, "corpus.txt")
    prefix = os.path.join(train_dir, "trained_corpus")
    vocab_size = 2048

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"❌ Training directory not found: {train_dir}")

    txt_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".txt")]
    if not txt_files:
        raise FileNotFoundError(f"❌ No .txt files found in {train_dir}")

    print(f"🚀 Found {len(txt_files)} training files. Combining into {corpus_path} ...")

    with open(corpus_path, "w", encoding="utf-8") as outfile:
        for fname in txt_files:
            print(f"  ➕ Adding {os.path.basename(fname)}")
            with open(fname, "r", encoding="utf-8") as infile:
                outfile.write(infile.read() + "\n")

    print(f"✅ Combined all training data into {corpus_path}")
    print(f"🚀 Training tokenizer from {corpus_path} ...")

    SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe"
    )

    print(f"✅ Tokenizer trained: {prefix}.model and {prefix}.vocab")

    # Export to binary format for C
    t = Tokenizer(f"{prefix}.model")
    t.export()
