from uuid import uuid4


def prepare_data(documents):
    # This function can be expanded to prepare data from various sources/formats.
    return documents

    
def create_document(session, document, verbosity=0):
    if not isinstance(document, dict) or "id" not in document or "text" not in document:
        if verbosity >= 1:
            print(f"Invalid document format: {document}")
        return  # or raise an exception
    session.run("""
        MERGE (d:Document {id: $id})
        SET d.text = $text
    """, id=document["id"], text=document["text"])


def create_sentence(session, document_id, sentence, verbosity=0):
    session.run("""
        MATCH (d:Document {id: $document_id})
        MERGE (s:Sentence {id: $sentence_id})
        SET s.text = $sentence_text
        MERGE (d)-[:HAS_SENTENCE]->(s)
    """, document_id=document_id, sentence_id=sentence["id"], sentence_text=sentence["text"])


def create_embedding(session, sentence_id, embedding_vector, verbosity=0):
    # Assuming embedding_vector is directly passed as the vector, and no separate ID is assigned to the embedding.
    # Generate a unique ID for embedding on-the-fly if needed, or simply link the embedding to the sentence.
    embedding_id = str(uuid4())  # If you decide to assign unique IDs to embeddings
    session.run("""
        MATCH (s:Sentence {id: $sentence_id})
        MERGE (e:Embedding {id: $embedding_id})
        SET e.vector = $embedding_vector
        MERGE (s)-[:HAS_EMBEDDING]->(e)
    """, sentence_id=sentence_id, embedding_id=embedding_id, embedding_vector=embedding_vector)


def create_kg(driver, documents, verbosity=0):
    prepared_documents = prepare_data(documents)
    with driver.session() as session:
        for doc in prepared_documents:
            if verbosity >= 2:
                print(f"Processing document: {doc['id']}")
            if isinstance(doc, dict):  # Ensure 'doc' is a dictionary
                create_document(session, doc, verbosity)
                for sent in doc.get("sentences", []):
                    create_sentence(session, doc["id"], sent, verbosity)
                    embedding_vector = sent.get("embeddings")  # Directly accessing the vector, adjust according to your actual structures
                    if embedding_vector:
                        create_embedding(session, sent["id"], embedding_vector, verbosity)


@data_exporter
def transform_custom(documents, *args, **kwargs):
    verbosity = kwargs.get('verbosity', 2)
    neo4j_driver, _postgres_conn = list(kwargs.get('factory_items_mapping').values())[0]

    arr = []
    for ds in documents:
        arr += ds

    create_kg(neo4j_driver, arr, verbosity)