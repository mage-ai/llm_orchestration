from pgvector.psycopg2 import register_vector


@custom
def transform_custom(*args, **kwargs):
    neo4j_driver, postgres_conn = list(kwargs.get('factory_items_mapping').values())[0]
    
    postgres_conn.autocommit = True
    cur = postgres_conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
    register_vector(postgres_conn)