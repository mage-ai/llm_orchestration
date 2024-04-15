import psycopg2
from neo4j import GraphDatabase
from pgvector.psycopg2 import register_vector


@factory
def graph(*args, **kwargs):
    neo4j_uri = 'neo4j://neo4j:7687'
    neo4j_user = 'neo4j'
    neo4j_password = 'password'
    neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    return neo4j_driver


@factory
def vector(*args, **kwargs):
    postgres_host = 'postgres'
    postgres_port = 5432
    postgres_db = 'llm_orchestration'
    postgres_user = 'postgres'
    postgres_password = 'password'

    postgres_conn = psycopg2.connect(
        host=postgres_host,
        port=postgres_port,
        dbname=postgres_db,
        user=postgres_user,
        password=postgres_password
    )

    return postgres_conn