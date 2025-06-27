import pandas as pd

df = pd.read_csv("hindi_wikipedia_texts.csv")
df = df[df["text"].str.len() > 100]  # optional filter
texts = df["text"].tolist()

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda")

print("🧠 Generating Hindi Wikipedia embeddings...")
wiki_embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

import faiss
import numpy as np
import os
import pickle

# Create index
wiki_embeddings = np.array(wiki_embeddings).astype("float32")
index = faiss.IndexFlatL2(wiki_embeddings.shape[1])
index.add(wiki_embeddings)

# Save index
os.makedirs("faiss_index", exist_ok=True)
faiss.write_index(index, "faiss_index/hindiwiki_index.idx")

# Save ID/text mapping
id_map = [f"wiki_{i}" for i in range(len(texts))]
with open("faiss_index/hindiwiki_id_map.pkl", "wb") as f:
    pickle.dump({"ids": id_map, "texts": texts}, f)

print("✅ Wikipedia FAISS index and ID map saved.")
