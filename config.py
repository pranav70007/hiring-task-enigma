# config_example.py — copy to config.py and fill with real values.
import os
from dotenv import load_dotenv
load_dotenv()  

NEO4J_URI = "bolt://api.neo4j.io"
NEO4J_USER = os.environ.get("NEO4J_CLIENT_ID")
NEO4J_PASSWORD = os.environ.get("NEO4J_CLIENT_SECRET")

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") 

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = "us-east1-gcp"   
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_VECTOR_DIM = 3072      