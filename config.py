# config_example.py — copy to config.py and fill with real values.
import os
from dotenv import load_dotenv
load_dotenv()  

NEO4J_URI = "neo4j+s://52203b73.databases.neo4j.io/db/neo4j/query/v2"#f"neo4j+s://{os.environ.get('CLIENT_NAME')}.databases.neo4j.io"
NEO4J_USER = "neo4j"#os.environ.get("CLIENT_ID")
NEO4J_PASSWORD = "r16vvMQvryaXYo-FrOaGZt6igHbCQvZabMzPV2w54aI"#os.environ.get("CLIENT_SECRET")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") 

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = "us-east1-gcp"   
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_VECTOR_DIM = 3072      