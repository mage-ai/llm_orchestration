def index_embeddings(conn, chunk, embeddings_for_tokens_for_chunk):
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_text TEXT PRIMARY KEY,
            vector VECTOR
        )
    """)

    cur.execute("""
        INSERT INTO embeddings (chunk_text, vector)
        VALUES (%s, %s)
        ON CONFLICT (chunk_text) DO UPDATE
        SET vector = EXCLUDED.vector;
    """, (chunk, embeddings_for_tokens_for_chunk))

    conn.commit()


@data_exporter
def export(data, *args, **kwargs):
    conn = list(kwargs.get('factory_items_mapping').values())[0][1]

    chunks_for_documents, tokens_for_chunks, embeddings_for_tokens_for_chunks_for_documents = data

    for chunks_for_document, embeddings_for_tokens_for_chunks in zip(chunks_for_documents, embeddings_for_tokens_for_chunks_for_documents):
        for chunk, embeddings_for_tokens_for_chunk in zip(chunks_for_document, embeddings_for_tokens_for_chunks):
            index_embeddings(conn, chunk, embeddings_for_tokens_for_chunk)
