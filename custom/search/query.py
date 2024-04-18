import ast  # For safely evaluating strings containing Python literals
import os

import faiss
import numpy as np
import pandas as pd


def create_faiss_index(conn):
    cur = conn.cursor()
    # Fetch stored embeddings, document IDs, and chunk texts
    cur.execute("SELECT chunk_text, document_id, vector FROM embeddings")
    rows = cur.fetchall()

    # Initialize lists
    chunk_texts = []
    document_ids = []
    vector_list = []

    for row in rows:
        chunk_texts.append(row[0])
        document_ids.append(row[1])
        
        # Safely evaluate the string representation of the list into an actual list of floats
        vector = np.array(ast.literal_eval(row[2]), dtype='float32')
        vector_list.append(vector)

    # Stack all vectors into a NumPy array
    vectors = np.vstack(vector_list)

    # Number of dimensions of each vector
    dim = vectors.shape[1]

    # Create a Faiss Index for L2 distance
    index = faiss.IndexFlatL2(dim)
    # Add vectors to the Faiss index
    index.add(vectors)

    return index, chunk_texts, document_ids


def create_query_vector(model, query: str) -> np.ndarray:
    """
    Encode the query string into a vector using the same model as for document embeddings.

    Parameters:
        query (str): The search query string.
    
    Returns:
        np.ndarray: The vector representation of the query.
    """
    query_embedding = model.encode(query)
    return query_embedding
    

def rag_query(driver, query_vector, faiss_index, chunk_texts, k=2):
    # Search for top-k most similar vectors in Faiss
    D, I = faiss_index.search(np.array([query_vector]).astype('float32'), 12)

    # Retrieve corresponding documents or chunks from Neo4j
    results = []

    with driver.session() as session:
        for i in range(k):
            obj = I[0]
            if i >= len(obj):
                i = -1
            index = obj[i]  # Index in the Faiss result to find the chunk index
            
            chunk_text = chunk_texts[index]  # Getting chunk_text using the index

            # Execute a query using chunk_text to find its document via the relationship in the graph
            result = session.run("""
            MATCH (d:Document)-[:CONTAINS]->(c:Chunk {text: $chunk_text})
            RETURN c.text AS chunk, d.document_id AS documentID
            """, chunk_text=chunk_text).single()

            if result:
                results.append(result)

    return results
    

@custom
def retrieve(*args, **kwargs):
    driver, postgres_conn = kwargs.get('factory_items_mapping')['database/drivers']
    model = kwargs.get('factory_items_mapping')['data_preparation/embeddings'][0]
    
    k_nearest_neighbors = kwargs.get('k_nearest_neighbors')
    query = kwargs.get('query')
    if not query:
        raise Exception('Query is required')

    query_vector = create_query_vector(model, query)
    index, chunk_texts, _document_ids = create_faiss_index(postgres_conn)
    nodes = rag_query(
        driver, 
        query_vector, 
        index, 
        chunk_texts, 
        k=k_nearest_neighbors,
    )

    arr = []
    for node in nodes:
        document_id = node['documentID']
        document = document_id
        
        if os.path.exists(document_id):
            with open(document_id, 'r') as f:
                document = f.read()

        arr.append(dict(
            chunk=node['chunk'],
            document=document,
            document_id=document_id,
        ))
    
    return arr