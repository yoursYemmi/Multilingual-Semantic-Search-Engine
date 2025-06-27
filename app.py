import streamlit as st
import numpy as np
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# -------------------------------
# 🔹 Load SentenceTransformer model
# -------------------------------
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)


# -------------------------------
# 🔹 Load TED FAISS index + metadata
# -------------------------------
ted_index = faiss.read_index("faiss_index/ted_crosslingual_5k.idx")
with open("faiss_index/id_map.pkl", "rb") as f:
    ted_meta = pickle.load(f)
ted_ids = ted_meta["ids"]
ted_texts = ted_meta["texts"]

# -------------------------------
# 🔹 Load Hindi Wikipedia index + metadata
# -------------------------------
wiki_index = faiss.read_index("faiss_index/hindiwiki_index.idx")
with open("faiss_index/hindiwiki_id_map.pkl", "rb") as f:
    wiki_meta = pickle.load(f)
wiki_ids = wiki_meta["ids"]
wiki_texts = wiki_meta["texts"]

# -------------------------------
# 🔹 Load English Wikipedia index + metadata
# -------------------------------
enwiki_index = faiss.read_index("faiss_index/englishwiki_index.idx")
with open("faiss_index/englishwiki_id_map.pkl", "rb") as f:
    enwiki_meta = pickle.load(f)
enwiki_ids = enwiki_meta["ids"]
enwiki_texts = enwiki_meta["texts"]

# -------------------------------
# 🖥️ Streamlit UI Setup
# -------------------------------
st.set_page_config(page_title=" Multilingual Semantic Search", layout="wide")
st.title("🌐 Multilingual Semantic Search Engine")

st.markdown("Search across:")
st.markdown("- 🟡 **TED Hindi–English Corpus**")
st.markdown("- 🟢 **Hindi Wikipedia**")
st.markdown("- 🔵 **English Wikipedia (India Topics)**")

query = st.text_input("🔍 Enter your query (Hindi or English):", "")

top_k = st.slider("📄 Number of results to show:", min_value=1, max_value=20, value=5)

# Source selector
search_source = st.multiselect(
    "🗂️ Choose datasets to search:",
    ["TED Corpus", "Hindi Wikipedia", "English Wikipedia"],
    default=["TED Corpus", "Hindi Wikipedia", "English Wikipedia"]
)

# Language filter
language_filter = st.selectbox("🗣️ Filter results by language:", ["Both", "Hindi only", "English only"])

# -------------------------------
# 🔍 Semantic Search Logic
# -------------------------------
if query:
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    all_results = []

    # 🔸 TED Corpus
    if "TED Corpus" in search_source:
        ted_distances, ted_indices = ted_index.search(query_embedding, 20)
        for i, dist in zip(ted_indices[0], ted_distances[0]):
            lang = "Hindi" if ted_ids[i].startswith("hi_") else "English"
            all_results.append({
                "source": "TED",
                "lang": lang,
                "text": ted_texts[i],
                "score": dist,
                "id": ted_ids[i]
            })

    # 🔸 Hindi Wikipedia
    if "Hindi Wikipedia" in search_source:
        wiki_distances, wiki_indices = wiki_index.search(query_embedding, 20)
        for i, dist in zip(wiki_indices[0], wiki_distances[0]):
            all_results.append({
                "source": "Hindi Wikipedia",
                "lang": "Hindi",
                "text": wiki_texts[i],
                "score": dist,
                "id": wiki_ids[i]
            })

    # 🔸 English Wikipedia
    if "English Wikipedia" in search_source:
        enwiki_distances, enwiki_indices = enwiki_index.search(query_embedding, 20)
        for i, dist in zip(enwiki_indices[0], enwiki_distances[0]):
            all_results.append({
                "source": "English Wikipedia",
                "lang": "English",
                "text": enwiki_texts[i],
                "score": dist,
                "id": enwiki_ids[i]
            })

    # Sort all results by similarity score (L2 distance)
    all_results.sort(key=lambda x: x["score"])

    # -------------------------------
    # 📋 Display Results
    # -------------------------------
    st.subheader("🔎 Top Semantic Matches:")

    shown = 0
    for item in all_results:
        if language_filter == "Hindi only" and item["lang"] != "Hindi":
            continue
        if language_filter == "English only" and item["lang"] != "English":
            continue

        shown += 1
        st.markdown(f"**{shown}. [{item['source']}] 🗣️ {item['lang']}**")
        st.write(item["text"])
        st.caption(f"📎 ID: {item['id']} | 🔍 Score (L2 distance): {item['score']:.4f}")

        if shown >= top_k:
            break

    if shown == 0:
        st.warning("⚠️ No matching results for selected filters.")
