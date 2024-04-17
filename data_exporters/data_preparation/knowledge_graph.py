from typing import List
from neo4j import Driver

def save_to_neo4j(driver: Driver, document_id: str, chunk_text: str, embeddings: List[float]):
    with driver.session() as session:
        # Assuming each call to this function deals with a single chunk of text,
        # and file_path acts as a unique identifier for the document.
        # The embeddings parameter represents embeddings for this single chunk.

        # Create or find a Document node based on the file_path (acting as a unique document identifier).
        session.write_transaction(lambda tx: tx.run(
            """
            MERGE (d:Document {document_id: $document_id})
            ON CREATE SET d.document_id = $document_id
            RETURN d
            """,
            document_id=document_id
        ))

        # Create a Chunk node with its embeddings and relate it to the Document node.
        session.write_transaction(lambda tx: tx.run(
            """
            MATCH (d:Document {document_id: $document_id})
            CREATE (c:Chunk {text: $chunk_text, embedding: $embedding})
            CREATE (d)-[:CONTAINS]->(c)
            """,
            document_id=document_id,
            chunk_text=chunk_text,
            embedding=embeddings  # Assuming embeddings is a list of floats; Neo4j supports list properties.
        ))

@data_exporter
def export(data, *args, **kwargs):
    driver = list(kwargs.get('factory_items_mapping').values())[0][0]

    chunks_for_documents, tokens_for_chunks, embeddings_for_tokens_for_chunks_for_documents = data

    for chunks_for_document, embeddings_for_tokens_for_chunks in zip(chunks_for_documents, embeddings_for_tokens_for_chunks_for_documents):
        file_path, cd_arr = chunks_for_document
        for chunk, embeddings_for_tokens_for_chunk in zip(cd_arr, embeddings_for_tokens_for_chunks[1]):
            save_to_neo4j(driver, file_path, chunk, embeddings_for_tokens_for_chunk)