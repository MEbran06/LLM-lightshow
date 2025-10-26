import sentencepiece as spm

# Put a representative corpus of your pairs here, with plenty of repetitions.
# Each line can be exactly the format you train on:
# <FRAME> LOW LOW LOW HIGH LOW LOW LOW LOW <SEP> 0 0 0 1 0 0 0 0 <EOT>
CORPUS = "dataset/train/corpus.txt"  # your combined train .txt

spm.SentencePieceTrainer.train(
    input=CORPUS,
    model_prefix="tokenizer",
    model_type="unigram",          # simple & robust for small vocabs
    vocab_size=22,                 # your chosen size
    unk_id=0,                      # keep <unk> at id=0
    bos_id=1, eos_id=2,            # optional, consistent with your training
    pad_id=-1,                     # no PAD if you do not use it
    byte_fallback=False,
    split_by_whitespace=True,
    treat_whitespace_as_suffix=False,
    allow_whitespace_only_pieces=False,
    hard_vocab_limit=False         # do not force rare pieces to <unk>
)
print("wrote tokenizer.model and tokenizer.vocab")
