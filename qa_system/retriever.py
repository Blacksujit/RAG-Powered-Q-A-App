from haystack.retriever.dense import DensePassageRetriever

def get_retriever(document_store):
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base"
    )
    # Update embeddings for documents
    document_store.update_embeddings(retriever)
    return retriever
