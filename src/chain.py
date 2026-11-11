"""Day 4: turn retrieved chunks into a grounded, cited answer.

Try it:  python -m src.chain "Is flood damage covered?"
"""
import sys

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src import config

SYSTEM_PROMPT = f"""You are an insurance policy assistant.
Answer the question using ONLY the policy excerpts in the context.
Rules:
- After each fact, cite its source like [policy-name p.3].
- Mention exclusions, limits or conditions if the context has them.
- If the context does not contain the answer, reply exactly: "{config.FALLBACK_ANSWER}"
- Never use outside knowledge and never guess.
- Keep the answer short and in plain English."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


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


def answer(question: str, policy: str | None = None, llm=None, docs: list[Document] | None = None) -> dict:
    """Retrieve, then generate. Pass `llm` or `docs` in tests to avoid API calls."""
    if docs is None:
        from src.retriever import retrieve

        docs = retrieve(question, policy=policy)

    if not docs:
        return {"answer": config.FALLBACK_ANSWER, "sources": []}

    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)

    reply = (PROMPT | llm).invoke({"context": format_context(docs), "question": question})
    text = reply.content.strip()

    # Hide sources when the model says it can't answer, so the UI doesn't show irrelevant pages.
    sources = [] if config.FALLBACK_ANSWER in text else to_sources(docs)
    return {"answer": text, "sources": sources}


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What does this policy cover?"
    result = answer(q)
    print(result["answer"])
    for s in result["sources"]:
        print(f"  - {s['policy']} p.{s['page']}")
