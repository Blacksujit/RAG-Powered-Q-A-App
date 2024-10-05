from .document_store import get_document_store
from .retriever import get_retriever
from .reader import get_reader
from haystack.pipeline import ExtractiveQAPipeline

# Set up document store, retriever, and reader
document_store = get_document_store()
retriever = get_retriever(document_store)
reader = get_reader()

# Initialize pipeline
pipeline = ExtractiveQAPipeline(reader, retriever)

def get_answer(question):
    # Run the pipeline
    prediction = pipeline.run(query=question, top_k_retriever=5, top_k_reader=3)
    answer = prediction['answers'][0].answer
    documents = [doc.content for doc in prediction['documents']]
    return answer, documents
