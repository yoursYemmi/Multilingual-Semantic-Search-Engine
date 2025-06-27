import os
import pandas as pd
from tqdm import tqdm


DATA_DIR = r"C:\Users\PRAVEEN\Downloads\train\train"

# Get list of all .txt files
files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".txt")])
print(f"Found {len(files)} .txt files")

#  Read files into memory
texts = []
ids = []

for file in tqdm(files, desc="📥 Reading text files"):
    try:
        file_path = os.path.join(DATA_DIR, file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                ids.append(file.replace(".txt", ""))
                texts.append(content)
    except Exception as e:
        print(f" Error reading {file}: {e}")

# Create DataFrame
df = pd.DataFrame({"id": ids, "text": texts})
print(f" Loaded {len(df)} documents into DataFrame")

df.to_csv("hindi_wikipedia_texts.csv", index=False, encoding="utf-8-sig")