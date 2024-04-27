from pgvector.psycopg2 import register_vector


@custom
def transform_custom(*args, **kwargs):
    _driver, conn = list(kwargs.get('factory_items_mapping').values())[0]
    
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
    register_vector(conn)