# Make changes in the data folder and run this once to upload the data on Pinecone

from dotenv import load_dotenv

load_dotenv()
import os
if "USER_AGENT" not in os.environ:
    os.environ["USER_AGENT"] = "MeridianBot/1.0"
from typing import List
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob
from db_config import get_vector_db

vector_db = get_vector_db()
# --- INGEST FILE (PDF / DOCX / MD) ---
def process_document(file_path: str) -> List:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".docx"]:
        loader = Docx2txtLoader(file_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # loader = UnstructuredFileLoader(
    #     file_path,
    #     strategy="fast",  # swap to "hi_res" for scanned PDFs needing OCR
    #     mode="elements",
    # )

    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, add_start_index=True
    )
    chunks = splitter.split_documents(raw_docs)

    for chunk in chunks:
        chunk.metadata["source"] = os.path.basename(file_path)

    return chunks
    # --- INGEST WEB URL ---


def ingest_web_url(url: str) -> str:
    # loader = UnstructuredURLLoader(urls=[url])
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    vector_db.add_documents(chunks)
    return f"Ingested URL: {url} ({len(chunks)} chunks)"
    # --- SYNC TO PINECONE ---


def sync_to_vector_db(chunks: List) -> None:
    vector_db.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks.")
    # --- ENTRYPOINT ---


if __name__ == "__main__":
    files = glob.glob("data/**/*", recursive=True)
    files = [f for f in files if os.path.isfile(f)]

    for file_path in files:
        try:
            chunks = process_document(file_path)
            sync_to_vector_db(chunks)
            print(f"From: {file_path}")
        except Exception as e:
            print(f"Failed {file_path}: {e}")
