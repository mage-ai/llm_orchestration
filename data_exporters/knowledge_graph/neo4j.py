import pandas as pd


@data_exporter
def export(df: pd.DataFrame, *args, **kwargs):
    factory_items_mapping = kwargs.get('factory_items_mapping')
    driver, _ = factory_items_mapping['database/drivers']

    reset = int(kwargs.get('reset', 1)) == 1

    print(f'df: {len(df)}')

    if reset:
        with driver.session() as session:
            print('Resetting...')
            session.write_transaction(lambda tx: tx.run('MATCH (n) DETACH DELETE n'))

    with driver.session() as session:
        for _index, row in df.iterrows():
            document_id = row['document_id']
            chunk = row['chunk']
            vector = row['vector']

            document_id = f'document{hash(document_id)}'

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
                    , embedding: $embedding
                    , text: $text
                })
                ON CREATE SET 
                    c.document_id = $document_id
                    , c.text = $text
                RETURN c
                """,
                document_id=document_id,
                embedding=vector,
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