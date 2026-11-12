"""Central settings. Change values here or override them in your .env file."""
import os

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "data")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = "policies"

# Free models: Groq for the LLM, a local Hugging Face model for embeddings.
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

# Tune these on Day 2 and Day 9 and record what changes.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "6"))

FALLBACK_ANSWER = "I can't find this in the policy."


def get_embeddings():
    """Embedding model used by both ingest and retrieval (they must match)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_llm():
    """Chat model that writes the answers. Reads GROQ_API_KEY from .env."""
    from langchain_groq import ChatGroq

    return ChatGroq(model=LLM_MODEL, temperature=0)
