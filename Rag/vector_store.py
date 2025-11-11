# rag/vector_store.py
"""
Chunk every CSV row into short sentences, embed with a lightweight
sentence-transformer, and store in a FAISS inner-product index.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

EMBEDDING_MODEL = "thenlper/gte-small"
CHUNK_WORDS = 50
OVERLAP = 0.5

def _chunk_sentence(text: str):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + CHUNK_WORDS])
        chunks.append(chunk)
        i += int(CHUNK_WORDS * (1 - OVERLAP))
    return chunks

def build_faiss_index(csv_path="data/sales.csv",
                     index_path="rag/faiss.index",
                     meta_path="rag/meta.pkl"):
    df = pd.read_csv(csv_path)

    encoder = SentenceTransformer(EMBEDDING_MODEL)
    all_chunks = []
    all_metadata = []

    for _, row in df.iterrows():
        description = (
            f"Date {row['date']} | Region {row['region']} | "
            f"Product {row['product']} | Units {row['units']} | Sales ${row['sales']}"
        )
        for chunk in _chunk_sentence(description):
            all_chunks.append(chunk)
            all_metadata.append(row.to_dict())

    vectors = encoder.encode(
        all_chunks, normalize_embeddings=True, batch_size=64, show_progress_bar=True
    )
    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors.astype("float32"))

    os.makedirs("rag", exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(all_metadata, f)

    print(f"FAISS index built – {len(all_chunks)} chunks")

if __name__ == "__main__":
    build_faiss_index()
