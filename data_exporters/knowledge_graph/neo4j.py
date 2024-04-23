from typing import List, Union


@data_exporter
def export(documents: List[List[Union[str, str, str, List[str], List[float]]]], *args, **kwargs):
    driver, conn = list(kwargs.get('factory_items_mapping').values())[0]

    reset = int(kwargs.get('reset', 1)) == 1

    documents_stored = []
    documents_failed = []

    if reset:
        with driver.session() as session:
            print('Resetting...')
            session.write_transaction(lambda tx: tx.run('MATCH (n) DETACH DELETE n'))

    with driver.session() as session:
        for source_document_id, _document, _metadata, text, _tokens, embedding_vector in documents:
            document_id = f'document{hash(source_document_id)}'

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
                embedding=embedding_vector,
                text=text,
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
                text=text,
                document_id=document_id,
                
            ))

            documents_stored.append(source_document_id)

            print(f'{len(documents_stored)}/{len(documents)}')

    print(f'documents_stored: {len(documents_stored)}')

    return [
        documents_stored,
        documents_failed,
    ]