from haystack.document_store.faiss import FAISSDocumentStore

def get_document_store():
    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat")
    documents = [
        {"content": "Django is a Python framework for web development."},
        {"content": "Flask is a micro web framework written in Python."}
    ]
    document_store.write_documents(documents)
    return document_store
