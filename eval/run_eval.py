"""Day 8-9: measure how good the assistant is.

Fill eval/questions.csv (aim for 50 rows), then run:
    python -m eval.run_eval --label baseline
Change one setting (CHUNK_SIZE, TOP_K, the prompt), re-run ingest, then:
    python -m eval.run_eval --label chunk500

Each run appends a summary row to eval/results_summary.csv, which becomes
the before/after table in your README.
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

from src import config
from src.chain import answer

EVAL_DIR = Path(__file__).parent

JUDGE_PROMPT = """You are grading an insurance assistant.
Question: {question}
Expected answer: {expected}
Assistant answer: {actual}
Does the assistant answer contain the same key facts as the expected answer,
with no contradictions? Reply with only YES or NO."""


def judge(llm, question: str, expected: str, actual: str) -> bool:
    reply = llm.invoke(JUDGE_PROMPT.format(question=question, expected=expected, actual=actual))
    return reply.content.strip().upper().startswith("YES")


def main() -> None:
    from langchain_openai import ChatOpenAI

    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run", help="Name for this experiment")
    parser.add_argument("--file", default=str(EVAL_DIR / "questions.csv"))
    args = parser.parse_args()

    judge_llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)
    rows = list(csv.DictReader(open(args.file, encoding="utf-8")))
    results = []

    for i, row in enumerate(rows, 1):
        result = answer(row["question"], policy=row.get("policy") or None)
        pages = {s["page"] for s in result["sources"]}
        is_oos = row["type"] == "out_of_scope"

        if is_oos:
            correct = config.FALLBACK_ANSWER in result["answer"]
            page_hit = None
        else:
            correct = judge(judge_llm, row["question"], row["expected_answer"], result["answer"])
            page_hit = int(row["expected_page"]) in pages if row.get("expected_page") else None

        results.append({**row, "actual_answer": result["answer"], "correct": correct, "page_hit": page_hit})
        print(f"[{i}/{len(rows)}] {'PASS' if correct else 'FAIL'}  {row['question']}")

    def rate(items):
        items = [x for x in items if x is not None]
        return round(100 * sum(items) / len(items), 1) if items else None

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "top_k": config.TOP_K,
        "questions": len(results),
        "accuracy_pct": rate([r["correct"] for r in results]),
        "page_hit_pct": rate([r["page_hit"] for r in results]),
        "refusal_pct": rate([r["correct"] for r in results if r["type"] == "out_of_scope"]),
    }

    detail_path = EVAL_DIR / f"results_{args.label}.csv"
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    summary_path = EVAL_DIR / "results_summary.csv"
    new_file = not summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(summary)

    print("\nSummary:", summary)
    print(f"Details saved to {detail_path}")


if __name__ == "__main__":
    main()
