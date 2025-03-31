import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

# Set up logging
logging.basicConfig(filename="logs/faiss_index.log", level=logging.INFO)

# Define paths
DATA_PATH = os.path.join("data", "qa_dataset.json")
INDEX_PATH = os.path.join("vector_database", "faiss_index.bin")
METADATA_PATH = os.path.join("vector_database", "contexts.json")

# Load dataset
try:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        logging.info("✅ QA dataset loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading dataset: {e}")
    raise e

# Extract contexts and compute embeddings
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
contexts = [item["context"] for item in data]
embeddings = embed_model.encode(contexts, convert_to_numpy=True)

# Save contexts for later retrieval
with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(contexts, f)

# Initialize FAISS Index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save FAISS index
faiss.write_index(index, INDEX_PATH)
logging.info("✅ FAISS index successfully created and saved!")

print("FAISS index successfully created!")
