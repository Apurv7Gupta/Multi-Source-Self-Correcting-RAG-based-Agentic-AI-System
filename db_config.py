import os
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


from dotenv import load_dotenv

load_dotenv()

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


# -------------To Delete DB----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Delete by ID
# index.delete(ids=["id1", "id2"])

# Delete by metadata filter
# index.delete(filter={"source": "company_report.md"})

# Delete ALL
# index.delete(delete_all=True)
# -----------------------------------------------------


# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") // not using Local embedding gen
