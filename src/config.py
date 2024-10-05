# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_PATH = os.path.join(BASE_DIR, "../data/documents.txt")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "../models/faiss_index.bin")

# Model names
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "facebook/bart-large-cnn"
