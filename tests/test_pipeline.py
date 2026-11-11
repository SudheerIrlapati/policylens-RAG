"""Day 6: unit tests. These run offline — no API key, no cost.

Run:  pytest -q
"""
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src import config
from src.chain import answer, format_context
from src.ingest import split_docs


def make_page(text, page=1, policy="home"):
    return Document(page_content=text, metadata={"policy": policy, "page_number": page})


def test_split_keeps_policy_and_page():
    long_text = "Coverage applies to the dwelling. " * 100
    chunks = split_docs([make_page(long_text, page=7)], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.metadata["page_number"] == 7 for c in chunks)
    assert all(c.metadata["policy"] == "home" for c in chunks)


def test_split_respects_chunk_size():
    chunks = split_docs([make_page("word " * 1000)], chunk_size=300, chunk_overlap=30)
    assert all(len(c.page_content) <= 300 for c in chunks)


def test_context_includes_citation_tags():
    ctx = format_context([make_page("Flood is excluded.", page=12)])
    assert "[home p.12]" in ctx
    assert "Flood is excluded." in ctx


def test_no_documents_returns_fallback():
    result = answer("Anything?", docs=[], llm=FakeListChatModel(responses=["should not be used"]))
    assert result["answer"] == config.FALLBACK_ANSWER
    assert result["sources"] == []


def test_answer_returns_sources():
    llm = FakeListChatModel(responses=["Flood damage is excluded [home p.12]."])
    result = answer("Is flood covered?", docs=[make_page("Flood is excluded.", page=12)], llm=llm)
    assert "excluded" in result["answer"]
    assert result["sources"][0]["page"] == 12


def test_fallback_answer_hides_sources():
    llm = FakeListChatModel(responses=[config.FALLBACK_ANSWER])
    result = answer("Car rental in Europe?", docs=[make_page("Dwelling coverage.")], llm=llm)
    assert result["sources"] == []
