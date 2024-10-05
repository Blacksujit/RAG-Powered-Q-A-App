# rag_pipeline.py
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from config import FAISS_INDEX_PATH, DATA_PATH

# Load FAISS index and models
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
retriever_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
generator_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
generator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def retrieve_docs(query, top_k=3):
    query_embedding = embed_text(query)
    _, indices = faiss_index.search(query_embedding, top_k)
    
    with open(DATA_PATH, "r") as file:
        documents = file.readlines()
    return [documents[i].strip() for i in indices[0]]

def generate_answer(context, query):
    input_text = f"question: {query} context: {context}"
    inputs = generator_tokenizer(input_text, return_tensors="pt")
    summary_ids = generator_model.generate(inputs["input_ids"], max_length=50, num_beams=5, early_stopping=True)
    return generator_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def answer_question(query):
    relevant_docs = retrieve_docs(query)
    context = " ".join(relevant_docs)
    return generate_answer(context, query)
