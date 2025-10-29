import os
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME       = os.getenv("INDEX_NAME", "mai-rag")
METRIC           = os.getenv("METRIC", "cosine")
CLOUD            = os.getenv("PINECONE_CLOUD", "aws")
REGION           = os.getenv("PINECONE_REGION", "eu-west-1")
DATA_DIR         = os.getenv("DATA_DIR", "data_md")
CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "200"))
