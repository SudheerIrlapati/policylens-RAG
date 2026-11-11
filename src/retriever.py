"""Day 3 and 5: search the vector database for chunks relevant to a question."""
from functools import lru_cache

from langchain_core.documents import Document

from src import config


@lru_cache(maxsize=1)
def get_vectorstore():
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=config.EMBED_MODEL),
        persist_directory=config.CHROMA_DIR,
    )


def retrieve(question: str, policy: str | None = None, k: int = config.TOP_K) -> list[Document]:
    """Return the top-k chunks for the question, optionally limited to one policy.

    Day 5 upgrade ideas: switch to MMR (vs.max_marginal_relevance_search),
    add BM25 hybrid search, or rerank the results.
    """
    vs = get_vectorstore()
    search_filter = {"policy": policy} if policy else None
    return vs.similarity_search(question, k=k, filter=search_filter)


def list_policies() -> list[str]:
    """Names of all indexed policies, for the UI dropdown."""
    metadatas = get_vectorstore().get(include=["metadatas"])["metadatas"]
    return sorted({m["policy"] for m in metadatas if m and "policy" in m})
