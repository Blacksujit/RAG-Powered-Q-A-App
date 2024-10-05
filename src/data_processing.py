# data_processing.py
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from config import DATA_PATH, FAISS_INDEX_PATH

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).numpy()
    return embeddings

def process_documents(file_path):
    embeddings = []
    with open(file_path, "r") as file:
        for line in file:
            embeddings.append(embed_text(line.strip()))
    return np.vstack(embeddings)

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

if __name__ == "__main__":
    embeddings = process_documents(DATA_PATH)
    create_faiss_index(embeddings)
