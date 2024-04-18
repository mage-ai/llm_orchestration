@data_exporter
def export_data(chunks_for_documents, tokens_for_chunks_for_documents, embeddings_for_tokens_for_chunks, **kwargs):
    chunks = []
    for chunks_for_document in chunks_for_documents:
        chunks += chunks_for_document

    tokens_for_chunks = []
    for tokens_for_chunks_for_document in tokens_for_chunks_for_documents:
        tokens_for_chunks += tokens_for_chunks_for_document

    embeddings_for_tokens = []
    for embeddings_for_tokens_for_chunk in embeddings_for_tokens_for_chunks:
        embeddings_for_tokens += embeddings_for_tokens_for_chunk

    return [
        chunks,
        tokens_for_chunks,
        embeddings_for_tokens,
    ]