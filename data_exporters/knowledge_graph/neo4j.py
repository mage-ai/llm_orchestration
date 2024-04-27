import json

import pandas as pd


@data_exporter
def export(df: pd.DataFrame, *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    driver, _ = factory_items_mapping['database/drivers']
    iterate = factory_items_mapping['helpers/lists'][0]

    reset = int(kwargs.get('reset', 1)) == 1
    if reset:
        with driver.session() as session:
            print('Resetting...')
            session.write_transaction(lambda tx: tx.run('MATCH (n) DETACH DELETE n'))

    with driver.session() as session:
        counter = 0

        for index, row in iterate(df):
            source_document_id = row['document_id']
            document_id = f'document{hash(source_document_id)}'
            
            print(f'{index}: {source_document_id}')
            
            chunk = row['chunk']
            vector = json.dumps(row['vector'])

            # Create or find a Document node
            session.write_transaction(lambda tx: tx.run(
                """
                MERGE (d:Document {document_id: $document_id})
                ON CREATE SET d.document_id = $document_id
                RETURN d
                """,
                document_id=document_id,
            ))

            # Create a Chunk node with its embeddings
            session.write_transaction(lambda tx: tx.run(
                """
                MERGE (c:Chunk {
                    document_id: $document_id
                    , embeddings: $embeddings
                    , text: $text
                })
                ON CREATE SET 
                    c.document_id = $document_id
                    , c.text = $text
                RETURN c
                """,
                document_id=document_id,
                embeddings=vector,
                text=chunk,
            ))

            # Relate Chunk to the Document node.
            session.write_transaction(lambda tx: tx.run(
                """
                MATCH (d:Document {
                    document_id: $document_id
                })
                MATCH (c:Chunk {
                    document_id: $document_id
                    , text: $text
                })
                CREATE (d)-[:CONTAINS]->(c)
                """,
                document_id=document_id,
                text=chunk,
            ))

            counter += 1
        
        print(f'items: {counter}')