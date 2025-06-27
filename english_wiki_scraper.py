indian_topics = [
    "India",
    "Indian economy",
    "Indian culture",
    "Indian education system",
    "Indian healthcare",
    "Indian politics",
    "Indian Constitution",
    "Indian history",
    "Indian agriculture",
    "Indian Space Research Organisation",
    "Indian Armed Forces",
    "Indian Railways",
    "Hindi language",
    "Swachh Bharat Abhiyan",
    "Digital India",
    "Indian sports",
    "Jee examination",
    "NEET examination"
]

import wikipedia
import pandas as pd

wikipedia.set_lang("en")  # English Wiki

texts = []
titles = []

for topic in indian_topics:
    try:
        content = wikipedia.page(topic).content
        texts.append(content)
        titles.append(topic)
    except Exception as e:
        print(f"❌ Skipped {topic}: {e}")

df = pd.DataFrame({"title": titles, "text": texts})
df.to_csv("english_wikipedia_india.csv", index=False, encoding="utf-8-sig")
print("✅ Saved English Wikipedia data.")


from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import os

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda")

df = pd.read_csv("english_wikipedia_india.csv")
texts = df["text"].tolist()

print("🔄 Generating embeddings...")
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

# Build index
embeddings = embeddings.astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save
os.makedirs("faiss_index", exist_ok=True)
faiss.write_index(index, "faiss_index/englishwiki_index.idx")
with open("faiss_index/englishwiki_id_map.pkl", "wb") as f:
    pickle.dump({"ids": [f"enwiki_{i}" for i in range(len(texts))], "texts": texts}, f)

print("✅ English Wikipedia FAISS index saved.")

