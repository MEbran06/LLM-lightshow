import os
from sentencepiece import SentencePieceProcessor

# --- SETTINGS ---
MODEL_PATH = "../dataset/validate/trained_corpus.model"
TEST_INPUT = "F 10 L0.200,0.150,0.090,0.020,0.010,0.009,0.010,0.006 ->"

# --- LOAD TOKENIZER ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at {MODEL_PATH}")
print(f"✅ Loaded tokenizer model from {MODEL_PATH}")

sp = SentencePieceProcessor(model_file=MODEL_PATH)

# --- ENCODE TEST INPUT ---
encoded = sp.encode(TEST_INPUT)
print(f"\n🔢 Encoded Input Tokens: {encoded[:20]} ... (len={len(encoded)})")

# --- SIMULATED MODEL OUTPUT ---
# In the real trained LLM model, this would come from inference
# For now, we just decode a continuation of the test pattern
simulated_output = encoded + [sp.piece_to_id("B0:ON"), sp.piece_to_id("B1:OFF"), sp.piece_to_id("B2:ON")]

decoded = sp.decode(simulated_output)
print("\n💡 Predicted LED Pattern:")
print(decoded)