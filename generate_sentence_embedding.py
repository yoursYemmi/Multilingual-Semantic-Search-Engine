import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os

# 🔹 Load your TED parallel data
df = pd.read_csv("ted_hi_en_parallel.csv")
df = df[:5000]  # ⏱ limit for quick testing

# 🔹 Load model on GPU
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device='cuda')

# 🔹 Generate Hindi embeddings
print(" Encoding Hindi...")
hindi_embeddings = model.encode(
    df["hindi"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

# 🔹 Generate English embeddings
print(" Encoding English...")
english_embeddings = model.encode(
    df["english"].tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

# 🔹 Save
os.makedirs("embeddings", exist_ok=True)
np.save("embeddings/hindi_embeddings_5k.npy", hindi_embeddings)
np.save("embeddings/english_embeddings_5k.npy", english_embeddings)
df.to_csv("embeddings/parallel_sentences_5k.csv", index=False, encoding="utf-8-sig")

print("✅ Done! Embeddings for 5,000 Hindi-English pairs saved.")
