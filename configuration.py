import os
import chromadb
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma

# --- EMBEDDINGS + CLIENT + VectorDB---
CHROMA_HOST = os.getenv("chroma_server", "localhost")
CHROMA_PORT = int(os.getenv("chroma_port", "8000"))
COLLECTION_NAME = "collection1"
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")


def get_vector_db() -> Chroma:
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    )
    client = chromadb.CloudClient(
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
        api_key=CHROMA_API_KEY,
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        client=client,
    )


# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") // not using Local embedding gen
# vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)    // not using local persistence

# Will probably be swapped with Pinecone (free tier) or Qdrant Cloud (free tier)
