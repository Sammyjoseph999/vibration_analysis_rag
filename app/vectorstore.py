"""Shared vector store settings, used for both indexing and retrieval."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

APP_DIR = Path(__file__).resolve().parent
PERSIST_DIR = APP_DIR / "agriculture_chromaV2"
COLLECTION_NAME = "agriculture"
EMBEDDING_MODEL = "text-embedding-3-large"


def get_embeddings():
    """Embedding model. Indexing and querying must use the same one."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file.")
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)


def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(PERSIST_DIR),
    )
