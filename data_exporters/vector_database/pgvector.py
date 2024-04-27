import hashlib
import json
from contextlib import closing

import pandas as pd


@data_exporter
def export(df: pd.DataFrame, *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    _, connection = factory_items_mapping['database/drivers']
    iterate = factory_items_mapping['helpers/lists'][0]
    
    error = None

    # This'll handle closing the connection
    with closing(connection) as conn:
        try:
            # This'll handle the transaction and closing the cursor
            with conn, conn.cursor() as cur:
                reset = int(kwargs.get('reset', 1)) == 1
                if reset:
                    print('Resetting...')
                    cur.execute('DROP TABLE IF EXISTS documents')

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        chunk_text_hash CHAR(64)
                        , document_id_hash CHAR(64)
                        , chunk_text TEXT
                        , document_id TEXT
                        , vector VECTOR
                        , embeddings JSONB
                        , metadata JSONB
                        , PRIMARY KEY (chunk_text_hash, document_id_hash)
                    );
                """)

                counter = 0
                for index, row in iterate(df):
                    source_document_id = row['document_id']
                    document_id = f'document{hash(source_document_id)}'
                    
                    print(f'{index}: {source_document_id}')
                    
                    chunk_text = row['chunk']
                    metadata = row['metadata'] or {}
                    metadata.update(dict(
                        page=0,
                        paragraph_id=0,
                        source_document_id=source_document_id,
                    ))
                    metadata_json = json.dumps(metadata)
                    
                    matrix = None
                    vector = row['vector']
                    if vector and isinstance(vector, list):
                        matrix = [[round(val, 10) for val in vector_row] for vector_row in vector]
                    else:
                        vector = [round(val, 10) for val in vector]

                    chunk_text_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
                    document_id_hash = hashlib.sha256(document_id.encode()).hexdigest()

                    cur.execute("""
                        INSERT INTO documents (
                            chunk_text_hash
                            , document_id_hash
                            , chunk_text
                            , document_id
                            , vector
                            , embeddings
                            , metadata
                        )
                        VALUES (
                            %s
                            , %s
                            , %s
                            , %s
                            , %s
                            , %s
                            , %s
                        )
                        ON CONFLICT (
                            chunk_text_hash
                            , document_id_hash
                        ) DO UPDATE
                        SET vector = EXCLUDED.vector;
                    """, (
                        chunk_text_hash,
                        document_id_hash,
                        chunk_text, 
                        document_id, 
                        vector if not matrix else None,
                        json.dumps(matrix) if matrix else None,
                        metadata_json,
                    ))

                    counter += 1

                conn.commit()

                print(f'items: {counter}')
        except Exception as err:
            error = err

    if error:
        raise error