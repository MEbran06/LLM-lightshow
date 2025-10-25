import os
import re
import shutil

def validate_corpus(corpus_path, validate_dir="dataset/validate"):
    validate_dir = os.path.abspath(validate_dir)
    print(f"🔍 Validating corpus file: {corpus_path}")
    if not os.path.exists(corpus_path):
        print(f"❌ Corpus file not found: {corpus_path}")
        return

    with open(corpus_path, "rb") as f:
        data = f.read()

    # Check structure
    pattern = rb"(MODEL|VOCAB|BIN):(\d+)\n"
    matches = list(re.finditer(pattern, data))

    if not matches:
        # No section headers found, check if file size > 1KB
        if len(data) > 1024:
            print("✅ Basic validation passed (raw .bin file detected)")
            # Create validate dir if needed
            os.makedirs(validate_dir, exist_ok=True)
            # Move corpus to validation folder
            dest_path = os.path.join(validate_dir, os.path.basename(corpus_path))
            dest_path = os.path.abspath(dest_path)
            shutil.copy2(corpus_path, dest_path)
            print(f"✅ Validation passed. Copied corpus to: {dest_path}")
            return
        else:
            print("⚠️ File too small to be a valid .bin file (less than 1KB).")
            return

    print(f"✅ Found {len(matches)} sections in corpus:")
    for m in matches:
        section, length = m.group(1).decode(), int(m.group(2))
        print(f"   • {section} ({length} bytes)")

    # Ensure all three core components exist
    required = {"MODEL", "VOCAB", "BIN"}
    found = {m.group(1).decode() for m in matches}
    if not required.issubset(found):
        print("❌ Missing required sections in corpus.")
        return

    # Create validate dir if needed
    os.makedirs(validate_dir, exist_ok=True)

    # Move corpus to validation folder
    dest_path = os.path.join(validate_dir, os.path.basename(corpus_path))
    dest_path = os.path.abspath(dest_path)
    shutil.copy2(corpus_path, dest_path)
    print(f"✅ Validation passed. Copied corpus to: {dest_path}")

if __name__ == "__main__":
    train_dir = "dataset/train"
    train_dir = os.path.abspath(train_dir)
    validate_dir = os.path.abspath("dataset/validate")
    print(f"🔍 Using train directory: {train_dir}")
    print(f"🔍 Using validate directory: {validate_dir}")
    print(f"🔍 Searching for latest .bin file in '{train_dir}'...")
    try:
        bin_files = [f for f in os.listdir(train_dir) if f.endswith(".bin")]
    except FileNotFoundError:
        print(f"❌ Training directory not found: {train_dir}")
        bin_files = []

    if not bin_files:
        print("❌ No .bin files found in training directory.")
    else:
        latest_file = max(bin_files, key=lambda f: os.path.getmtime(os.path.join(train_dir, f)))
        latest_path = os.path.abspath(os.path.join(train_dir, latest_file))
        print(f"✅ Latest .bin file found: {latest_file}")
        validate_corpus(latest_path, validate_dir)