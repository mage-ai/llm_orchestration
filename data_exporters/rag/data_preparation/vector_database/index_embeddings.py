@data_exporter
def index_embeddings(documents, *args, **kwargs):
    _neo4j_driver, postgres_conn = list(kwargs.get('factory_items_mapping').values())[0]
    verbosity = kwargs.get('verbosity', 1)  # Added verbosity support

    cur = postgres_conn.cursor()
    
    if verbosity >= 1:
        print("Ensuring pgvector extension is available and embeddings table exists...")
        
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            sentence_id TEXT PRIMARY KEY,
            vector vector(1536)
        );
    """)

    arr = []
    for ds in documents:
        arr += ds
    
    processed_sentences = 0
    total_sentences = sum(len(doc.get("sentences", [])) for doc in arr)
    
    if verbosity >= 1:
        print(f"Total number of sentences to index: {total_sentences}")

    for doc in arr:
        for sentence in doc.get("sentences", []):
            if verbosity >= 2:
                print('Current sentence:', sentence)
                
            # Directly access the embeddings vector, assuming it's directly stored under 'embeddings'
            vector = sentence.get("embeddings")
            if vector:
                # Prepare the vector string representation for SQL insertion using square brackets
                vector_data = "[" + ",".join([str(v) for v in vector]) + "]"
                
                # Assuming 'sentence_id' is directly accessible in the 'sentence' dictionary
                sentence_id = sentence.get("id")
                if sentence_id:
                    cur.execute("""
                        INSERT INTO embeddings (sentence_id, vector)
                        VALUES (%s, vector(%s))
                        ON CONFLICT (sentence_id) DO UPDATE
                        SET vector = EXCLUDED.vector;
                    """, (sentence_id, vector_data))
                    processed_sentences += 1
                    if verbosity >= 1:
                        print(f"Processed {processed_sentences}/{total_sentences} sentences. Last processed sentence_id: {sentence_id}")

    if verbosity >= 0:
        print("Embeddings indexation complete.")

    postgres_conn.commit()