"""Day 4: turn retrieved chunks into a grounded, cited answer.

Try it:  python -m src.chain "Is flood damage covered?"
"""
import re
import sys

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src import config

SYSTEM_PROMPT = f"""You are an insurance policy assistant.
Answer the question using ONLY the policy excerpts in the context.
Rules:
- After each fact, cite its source like [policy-name p.3].
- Mention exclusions, limits or conditions if the context has them.
- If the excerpts come from more than one policy, answer for each policy separately and name it. Never combine numbers from different policies.
- If the context does not contain the answer, reply exactly: "{config.FALLBACK_ANSWER}"
- Never use outside knowledge and never guess.
- Keep the answer short and in plain English."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

# Matches the text inside [ ... ] or 【 ... 】 (the model uses both styles).
_BRACKET_RE = re.compile(r"[\[【]([^\]】]+)[\]】]")
_POLICY_RE = re.compile(r"\s*([A-Za-z0-9_\-]+)")
_PAGE_RE = re.compile(r"p\.?\s*(\d+)")


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('policy', 'unknown')} p.{d.metadata.get('page_number', '?')}]\n{d.page_content}"
        for d in docs
    )


def to_sources(docs: list[Document]) -> list[dict]:
    return [
        {
            "policy": d.metadata.get("policy"),
            "page": d.metadata.get("page_number"),
            "snippet": d.page_content[:300],
        }
        for d in docs
    ]


def extract_citations(text: str) -> set[tuple[str, int]]:
    """Find every (policy, page) the answer cites.

    Handles [star_health p.31], [ star_health p.31 ], 【star_health p.31】
    and several pages in one bracket like [bajaj_allianz_health p.6, p.15].
    """
    found: set[tuple[str, int]] = set()
    for inner in _BRACKET_RE.findall(text):
        policy = _POLICY_RE.match(inner)
        pages = _PAGE_RE.findall(inner)
        if policy and pages:
            for page in pages:
                found.add((policy.group(1), int(page)))
    return found


def check_citations(text: str, docs: list[Document]) -> list[str]:
    """Return citations in the answer that do NOT match any retrieved chunk (likely made up)."""
    retrieved = {(d.metadata.get("policy"), d.metadata.get("page_number")) for d in docs}
    return sorted(f"{policy} p.{page}" for policy, page in extract_citations(text) if (policy, page) not in retrieved)


def answer(question: str, policy: str | None = None, llm=None, docs: list[Document] | None = None) -> dict:
    """Retrieve, then generate. Pass `llm` or `docs` in tests to avoid API calls."""
    if docs is None:
        from src.retriever import retrieve

        docs = retrieve(question, policy=policy)

    if not docs:
        return {"answer": config.FALLBACK_ANSWER, "sources": [], "invalid_citations": []}

    if llm is None:
        llm = config.get_llm()

    reply = (PROMPT | llm).invoke({"context": format_context(docs), "question": question})
    text = reply.content.strip()

    # Hide sources when the model says it can't answer, so the UI doesn't show irrelevant pages.
    sources = [] if config.FALLBACK_ANSWER in text else to_sources(docs)
    return {"answer": text, "sources": sources, "invalid_citations": check_citations(text, docs)}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What does this policy cover?"
    result = answer(q)
    print(result["answer"])
    for s in result["sources"]:
        print(f"  - {s['policy']} p.{s['page']}")
    if result["invalid_citations"]:
        print("WARNING: cited pages not in the retrieved chunks:", ", ".join(result["invalid_citations"]))