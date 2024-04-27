def index_embeddings(conn, document_id, chunk, embeddings_for_tokens_for_chunk):
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
    """, (chunk, document_id, embeddings_for_tokens_for_chunk))

    conn.commit()


@data_exporter
def export(data, *args, **kwargs):
    conn = list(kwargs.get('factory_items_mapping').values())[0][1]

    for chunks, _tokens, embeddings_for_tokens in zip(*data):
        document_id = chunks[1]

        for chunk, embeddings in zip(chunks[0], embeddings_for_tokens[0]):
            index_embeddings(
                conn, 
                document_id,
                chunk,
                embeddings,
            )