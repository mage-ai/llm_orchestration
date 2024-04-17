import os

import psycopg2
from neo4j import GraphDatabase
from pgvector.psycopg2 import register_vector

from mage_ai.data_preparation.shared.secrets import get_secret_value


@factory
def graph(*args, **kwargs):
    neo4j_uri = get_secret_value('NEO4J_URI') or \
        os.getenv('NEO4J_URI', 'neo4j://neo4j:7687')
    neo4j_user = get_secret_value('NEO4J_USERNAME') or \
        os.getenv('NEO4J_USERNAME', 'neo4j')
    neo4j_password = get_secret_value('NEO4J_PASSWORD') or \
        os.getenv('NEO4J_PASSWORD', 'password')    
    neo4j_driver = GraphDatabase.driver(
        neo4j_uri, 
        auth=(neo4j_user, neo4j_password),
        
    )

    return neo4j_driver


@factory
def vector(*args, **kwargs):
    postgres_host = 'postgres'
    postgres_port = 5432
    postgres_db = os.getenv('POSTGRES_DB', 'llm_orchestration')
    postgres_password = os.getenv('POSTGRES_PASSWORD', 'password')
    postgres_user = os.getenv('POSTGRES_USER', 'postgres')

    postgres_conn = psycopg2.connect(
        host=postgres_host,
        port=postgres_port,
        dbname=postgres_db,
        user=postgres_user,
        password=postgres_password
    )

    return postgres_conn