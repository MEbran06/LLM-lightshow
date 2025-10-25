import os
import subprocess

def run_command(cmd_list, desc):
    print(f"\n⚙️  {desc} ...")
    result = subprocess.run(cmd_list, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error while running {desc}:")
        print(result.stderr)
        exit(1)
    print(f"✅ {desc} completed successfully.")
    return result

# 1️⃣ Step 1: Generate features and commands
run_command([
    "python3", "make_dataset.py",
    "--songs_dir", "songs",
    "--out_dir", "dataset/test",
    "--avg_n", "1",
    "--sensitivity", "0.05"
], "Generating dataset features and commands")

# 2️⃣ Step 2: Combine LED dataset
run_command(["python3", "combine_led_dataset.py"], "Combining LED dataset")

# 3️⃣ Step 3: Move combined data to train folder
os.makedirs("dataset/train", exist_ok=True)
os.system("mv dataset/test/*_trained.txt dataset/train/ 2>/dev/null || true")

# 4️⃣ Step 4: Train tokenizer
run_command([
    "python3", "llama2.c/tokenizer.py",
    "-t", "dataset/train/trained_001.txt"
], "Training tokenizer")

# 5️⃣ Step 5: Combine corpus
run_command(["python3", "llama2.c/combine_corpus.py"], "Combining corpus")

# 6️⃣ Step 6: Validate corpus
run_command(["python3", "llama2.c/validate_corpus.py"], "Validating corpus")

print("\n🚀 Auto-train pipeline completed successfully! All files validated and ready.")