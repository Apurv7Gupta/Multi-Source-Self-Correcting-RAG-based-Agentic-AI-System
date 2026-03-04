import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


# --- EMBEDDINGS + CLIENT + VectorDB---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "collection1")


def get_vector_db() -> PineconeVectorStore:
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    )
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    return PineconeVectorStore(index=index, embedding=embeddings)


# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") // not using Local embedding gen
# vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)    // not using local persistence

# Will probably be swapped with Pinecone (free tier) or Qdrant Cloud (free tier)
