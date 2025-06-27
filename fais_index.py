import numpy as np
import pandas as pd
import faiss
import os
import pickle

# Load embeddings
hindi_embeddings = np.load("embeddings/hindi_embeddings_5k.npy").astype("float32")
english_embeddings = np.load("embeddings/english_embeddings_5k.npy").astype("float32")

# Load corresponding sentence text
df = pd.read_csv("embeddings/parallel_sentences_5k.csv")

# Combine embeddings into one array
all_embeddings = np.vstack([hindi_embeddings, english_embeddings])

# Create ID mapping for back-reference
id_map = (
    ["hi_" + str(i) for i in range(len(hindi_embeddings))] +
    ["en_" + str(i) for i in range(len(english_embeddings))]
)

# Save sentence metadata (id to sentence text)
metadata = id_map
texts = df["hindi"].tolist() + df["english"].tolist()

os.makedirs("faiss_index", exist_ok=True)

# Save metadata
with open("faiss_index/id_map.pkl", "wb") as f:
    pickle.dump({"ids": metadata, "texts": texts}, f)

# Build FAISS index
dimension = all_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # use L2 distance
index.add(all_embeddings)

# Save index
faiss.write_index(index, "faiss_index/ted_crosslingual_5k.idx")

print(f" FAISS index built and saved with {index.ntotal} vectors.")
