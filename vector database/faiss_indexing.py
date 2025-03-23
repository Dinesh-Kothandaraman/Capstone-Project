from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Load dataset
with open(r"C:\Users\dines\OneDrive\Documents\GitHub\Capstone Project\qa_dataset.json", "r") as f:
    data = json.load(f)

# Extract contexts and compute embeddings
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
contexts = [item["context"] for item in data]
embeddings = embed_model.encode(contexts, convert_to_numpy=True)

# Index embeddings with FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, "faiss_index.bin")

print("FAISS index successfully created!")
