def create_nodes_and_relationships(driver, chunks, tokenized_chunks, embeddings):
    with driver.session() as session:
        for i, chunk in enumerate(chunks):
            # First, add a chunk node
            session.run("MERGE (c:Chunk {text: $chunkText})", chunkText=chunk)

            # For each sentence in the tokenized chunk, add a sentence node,
            # then link it to the chunk, and to its corresponding embedding.
            for j, sentence in enumerate(tokenized_chunks[i]):
                sentenceText = sentence
                embeddingVector = embeddings[i][j]  # Access embedding based on indices
                
                # Transactions to create Sentence and Embedding nodes, and relations
                session.run("""
                MATCH (c:Chunk {text: $chunkText})
                MERGE (s:Sentence {text: $sentenceText})
                MERGE (c)-[:CONTAINS]->(s)
                
                // For every sentence, create an Embedding with a unique ID
                // Here, using a combination of chunk and sentence indices to form an ID
                MERGE (e:Embedding {id: $embeddingId})
                ON CREATE SET e.vector = $vector
                
                MERGE (s)-[:HAS_EMBEDDING]->(e)
                """, 
                chunkText=chunk, 
                sentenceText=sentenceText, 
                embeddingId=f"emb_{i}_{j}", 
                vector=embeddingVector)


@data_exporter
def export(data, *args, **kwargs):
    neo4j_driver = list(kwargs.get('factory_items_mapping').values())[0][0]
    # Level 0 index: 
        # Index 0: chunk
        # Index 1: tokenized_chunk
        # Index 2: embedding
    # Level 1 index:
        # Index 0: Document 1
        # Index 1: Document 2
        # Index 2: Document 3
    # Level 2 index:
        # Index 0: only

    chunks_for_documents, tokens_for_chunks, embeddings_for_tokens = data

    # create_nodes_and_relationships(neo4j_driver, chunks, tokens_for_chunks, embeddings_for_tokens)