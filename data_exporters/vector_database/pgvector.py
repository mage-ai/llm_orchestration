import hashlib
import json
from contextlib import closing

import pandas as pd


@data_exporter
def export(df: pd.DataFrame, *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    _, connection = factory_items_mapping['database/drivers']
    
    print(f'df: {len(df)}')

    error = None
    # This'll handle closing the connection
    with closing(connection) as conn:
        try:
            # This'll handle the transaction and closing the cursor
            with conn, conn.cursor() as cur:
                reset = int(kwargs.get('reset', 1)) == 1
                if reset:
                    print('Resetting...')
                    cur.execute('DROP TABLE IF EXISTS embeddings')

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        chunk_text_hash CHAR(64)
                        , document_id_hash CHAR(64)
                        , chunk_text TEXT
                        , document_id TEXT
                        , vector VECTOR
                        , metadata JSONB
                        , PRIMARY KEY (chunk_text_hash, document_id_hash)
                    );
                """)

                counter = 0
                for _index, row in df.iterrows():
                    source_document_id = row['document_id']
                    document_id = f'document{hash(source_document_id)}'
                    chunk_text = row['chunk']
                    metadata = row['metadata']
                    vector = row['vector']

                    print(source_document_id)
                    document_id = f'document{hash(source_document_id)}'

                    metadata = metadata or {}
                    metadata.update(dict(
                        page=0,
                        paragraph_id=0,
                        source_document_id=source_document_id,
                    ))
                    metadata_json = json.dumps(metadata)

                    chunk_text_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
                    document_id_hash = hashlib.sha256(document_id.encode()).hexdigest()

                    cur.execute("""
                        INSERT INTO embeddings (
                            chunk_text_hash
                            , document_id_hash
                            , chunk_text
                            , document_id
                            , vector
                            , metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
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
                        [round(val, 10) for val in vector],
                        metadata_json,
                    ))

                    counter += 1
                    
                    print(f'{counter}/{len(df)})')

                conn.commit()
        except Exception as err:
            error = err

    if error:
        raise error