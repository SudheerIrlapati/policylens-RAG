"""Milestone 4: offline tests for citation checking and the policy filter (no API calls, no cost)."""
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.chain import answer, check_citations, extract_citations

DOCS = [
    Document(
        page_content="PED excluded until the expiry of 36 months of continuous coverage.",
        metadata={"policy": "star_health", "page_number": 31},
    ),
    Document(
        page_content="The PED waiting period as opted would be specified on the Policy Schedule.",
        metadata={"policy": "bajaj_allianz_health", "page_number": 15},
    ),
]


def test_extract_handles_all_bracket_styles():
    text = (
        "A [star_health p.31]. B [ hdfc_ergo_home p.2 ]. "
        "C 【star_health p.6】. D [bajaj_allianz_health p.6, p.15]."
    )
    assert extract_citations(text) == {
        ("star_health", 31),
        ("hdfc_ergo_home", 2),
        ("star_health", 6),
        ("bajaj_allianz_health", 6),
        ("bajaj_allianz_health", 15),
    }


def test_extract_ignores_brackets_without_a_page():
    assert extract_citations("See [Section 4] and [this link](https://example.com).") == set()


def test_real_citations_pass():
    text = "Star: 36 months [star_health p.31]. Bajaj: as opted 【bajaj_allianz_health p.15】."
    assert check_citations(text, DOCS) == []


def test_made_up_page_is_flagged():
    # The real bug we saw: gpt-oss-20b cited p.45 (a definition number), which was never retrieved.
    assert check_citations("Up to 24 months [bajaj_allianz_health p.45].", DOCS) == ["bajaj_allianz_health p.45"]


def test_answer_reports_invalid_citations():
    fake_llm = FakeListChatModel(responses=["Up to 24 months [bajaj_allianz_health p.45]."])
    result = answer("What is the PED waiting period?", llm=fake_llm, docs=DOCS)
    assert result["invalid_citations"] == ["bajaj_allianz_health p.45"]


def test_policy_filter_only_returns_that_policy(monkeypatch):
    from langchain_chroma import Chroma
    from langchain_core.embeddings import DeterministicFakeEmbedding

    import src.retriever as retriever

    docs = [
        Document(page_content=f"waiting period clause {i}", metadata={"policy": policy, "page_number": i})
        for i, policy in enumerate(["star_health", "bajaj_allianz_health"] * 5, start=1)
    ]
    store = Chroma.from_documents(
        docs, embedding=DeterministicFakeEmbedding(size=32), collection_name="test_policy_filter"
    )
    monkeypatch.setattr(retriever, "get_vectorstore", lambda: store)

    results = retriever.retrieve("waiting period", policy="star_health", k=3)
    assert results
    assert all(d.metadata["policy"] == "star_health" for d in results)