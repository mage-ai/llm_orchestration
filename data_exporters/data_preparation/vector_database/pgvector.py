def index_embeddings(conn, file_path, chunk, embeddings_for_tokens_for_chunk):
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_text TEXT PRIMARY KEY,
            document_id TEXT,
            vector VECTOR
        )
    """)

    cur.execute("""
        INSERT INTO embeddings (chunk_text, document_id, vector)
        VALUES (%s, %s, %s)
        ON CONFLICT (chunk_text) DO UPDATE
        SET vector = EXCLUDED.vector;
    """, (chunk, file_path, embeddings_for_tokens_for_chunk))

    conn.commit()


@data_exporter
def export(data, *args, **kwargs):
    conn = list(kwargs.get('factory_items_mapping').values())[0][1]

    chunks_for_documents, tokens_for_chunks, embeddings_for_tokens_for_chunks_for_documents = data

    for chunks_for_document, embeddings_for_tokens_for_chunks in zip(chunks_for_documents, embeddings_for_tokens_for_chunks_for_documents):
        file_path, cd_arr = chunks_for_document
        for chunk, embeddings_for_tokens_for_chunk in zip(cd_arr, embeddings_for_tokens_for_chunks[1]):
            index_embeddings(conn, file_path, chunk, embeddings_for_tokens_for_chunk)