"""Compare vector retrieval with cross-encoder reranking."""

import json
import time
from pathlib import Path

from sentence_transformers import CrossEncoder

from app.answerability_evaluator import evaluate_answerability_retrieval
from app.chunker import chunk_document
from app.document_loader import load_documents
from app.embedder import create_embedder, embed_chunks
from app.ranking_evaluator import reciprocal_rank
from app.reranker import rerank_chunks
from app.vector_search import search_chunks


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = PROJECT_ROOT / "evals" / "questions.json"
RANKING_QUESTIONS_FILE = PROJECT_ROOT / "evals" / "ranking_questions.json"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_COUNT = 10
FINAL_TOP_K = 3
MIN_SCORE = -1.0


def load_json_list(path: Path) -> list[dict]:
    """Load and validate a JSON list of dictionaries."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"{path.name} must contain a JSON list.")

    if not all(isinstance(item, dict) for item in data):
        raise ValueError(f"Every entry in {path.name} must be an object.")

    return data


def load_questions() -> list[dict]:
    """Load document-level evaluation questions."""
    questions = load_json_list(QUESTIONS_FILE)
    required = {"id", "question", "expected_documents", "answerable"}

    for index, question in enumerate(questions):
        missing = required - question.keys()
        if missing:
            raise ValueError(
                f"Question at index {index} is missing fields: "
                f"{sorted(missing)}"
            )

        if not isinstance(question["answerable"], bool):
            raise ValueError(
                f"Question {question['id']} must have a boolean "
                "'answerable' field."
            )

        if not isinstance(question["expected_documents"], list):
            raise ValueError(
                f"Question {question['id']} must have an "
                "'expected_documents' list."
            )

    return questions


def load_ranking_questions() -> dict[str, list[dict[str, str]]]:
    """Load section-level relevance labels indexed by question ID."""
    records = load_json_list(RANKING_QUESTIONS_FILE)
    labels = {}

    for index, record in enumerate(records):
        if "id" not in record or "relevant_sections" not in record:
            raise ValueError(
                f"Ranking record at index {index} requires 'id' and "
                "'relevant_sections'."
            )

        question_id = record["id"]

        if question_id in labels:
            raise ValueError(
                f"Duplicate ranking question ID: {question_id}"
            )

        relevant_sections = record["relevant_sections"]

        if not isinstance(relevant_sections, list) or not relevant_sections:
            raise ValueError(
                f"Question {question_id} must have a non-empty "
                "'relevant_sections' list."
            )

        for section in relevant_sections:
            if (
                not isinstance(section, dict)
                or not isinstance(section.get("source"), str)
                or not isinstance(section.get("section"), str)
            ):
                raise ValueError(
                    f"Invalid relevant section for question {question_id}."
                )

        labels[question_id] = relevant_sections

    return labels


def build_knowledge_base() -> list[dict]:
    """Load documents and split them into chunks."""
    chunks = []

    for document in load_documents():
        chunks.extend(chunk_document(document))

    if not chunks:
        raise RuntimeError("No knowledge-base chunks were created.")

    return chunks


def evaluate_document_retrieval(
    questions: list[dict],
    retrieved_results: list[list[dict]],
) -> tuple[dict, list[dict]]:
    """Evaluate document-level retrieval and answerability."""
    question_results = []

    for question, chunks in zip(questions, retrieved_results):
        retrieved_documents = sorted({chunk["source"] for chunk in chunks})

        question_results.append(
            {
                "id": question["id"],
                "question": question["question"],
                "answerable": question["answerable"],
                "expected_documents": question["expected_documents"],
                "retrieved_documents": retrieved_documents,
            }
        )

    return (
        evaluate_answerability_retrieval(question_results),
        question_results,
    )


def print_document_summary(label: str, summary: dict) -> None:
    """Print document retrieval and answerability metrics."""
    print(f"\n{label}")
    print("-" * 60)

    success = summary["answerable_retrieval_success_rate"]
    unanswerable = summary["unanswerable_retrieval_rate"]

    print(
        f"Answerable retrieval success: {success:.1%}"
        if success is not None
        else "Answerable retrieval success: N/A"
    )
    print(
        f"Unanswerable retrieval rate: {unanswerable:.1%}"
        if unanswerable is not None
        else "Unanswerable retrieval rate: N/A"
    )


def main() -> None:
    questions = load_questions()
    ranking_labels = load_ranking_questions()

    if not questions:
        print("No evaluation questions were found.")
        return

    question_by_id = {question["id"]: question for question in questions}
    unknown_ids = set(ranking_labels) - set(question_by_id)

    if unknown_ids:
        raise ValueError(
            f"Ranking labels reference unknown question IDs: "
            f"{sorted(unknown_ids)}"
        )

    ranking_ids = set(ranking_labels)
    ranking_questions = [
        question for question in questions if question["id"] in ranking_ids
    ]

    print(f"Document-level evaluation questions: {len(questions)}")
    print(f"Section-level ranking questions: {len(ranking_questions)}")

    chunks = build_knowledge_base()
    print(f"Knowledge-base chunks: {len(chunks)}")

    print("\nLoading embedding model...")
    embedding_model = create_embedder()

    print("Embedding knowledge-base chunks...")
    embedded_chunks = embed_chunks(chunks, embedding_model)

    print("Loading cross-encoder...")
    reranker_model = CrossEncoder(RERANKER_MODEL)

    baseline_results = {}
    reranked_results = {}
    reranking_times = {}

    print("\nEvaluating baseline and reranked retrieval...")

    for question in questions:
        candidates = search_chunks(
            query=question["question"],
            embedded_chunks=embedded_chunks,
            model=embedding_model,
            top_k=CANDIDATE_COUNT,
            min_score=MIN_SCORE,
        )

        baseline_results[question["id"]] = candidates[:FINAL_TOP_K]

        start = time.perf_counter()
        reranked_results[question["id"]] = rerank_chunks(
            query=question["question"],
            chunks=candidates,
            model=reranker_model,
            top_k=FINAL_TOP_K,
        )
        reranking_times[question["id"]] = time.perf_counter() - start

    baseline_summary, _ = evaluate_document_retrieval(
        questions,
        [baseline_results[q["id"]] for q in questions],
    )
    reranked_summary, _ = evaluate_document_retrieval(
        questions,
        [reranked_results[q["id"]] for q in questions],
    )

    print_document_summary("BASELINE: VECTOR SEARCH TOP 3", baseline_summary)
    print_document_summary("RERANKED: CROSS-ENCODER TOP 3", reranked_summary)

    baseline_ranks = {}
    reranked_ranks = {}

    for question in ranking_questions:
        question_id = question["id"]
        relevant = ranking_labels[question_id]

        baseline_ranks[question_id] = reciprocal_rank(
            baseline_results[question_id], relevant
        )
        reranked_ranks[question_id] = reciprocal_rank(
            reranked_results[question_id], relevant
        )

    baseline_mrr = sum(baseline_ranks.values()) / len(baseline_ranks)
    reranked_mrr = sum(reranked_ranks.values()) / len(reranked_ranks)

    print("\nSECTION-LEVEL RANKING EVALUATION")
    print("-" * 60)
    print(f"Questions evaluated: {len(ranking_questions)}")
    print(f"Baseline MRR: {baseline_mrr:.4f}")
    print(f"Reranked MRR: {reranked_mrr:.4f}")
    print(f"MRR change: {reranked_mrr - baseline_mrr:+.4f}")

    print("\nPER-QUESTION RANKING COMPARISON")
    print("-" * 60)

    improved = 0
    unchanged = 0
    regressed = 0

    for question in ranking_questions:
        question_id = question["id"]
        baseline_rank = baseline_ranks[question_id]
        reranked_rank = reranked_ranks[question_id]

        if reranked_rank > baseline_rank:
            outcome = "IMPROVED"
            improved += 1
        elif reranked_rank < baseline_rank:
            outcome = "REGRESSED"
            regressed += 1
        else:
            outcome = "UNCHANGED"
            unchanged += 1

        print(
            f"{question_id}: baseline RR={baseline_rank:.3f}, "
            f"reranked RR={reranked_rank:.3f} | {outcome}"
        )

    order_changes = sum(
        [
            [
                (chunk["source"], chunk["section"])
                for chunk in baseline_results[q["id"]]
            ]
            != [
                (chunk["source"], chunk["section"])
                for chunk in reranked_results[q["id"]]
            ]
            for q in questions
        ]
    )

    print("\nFINAL COMPARISON")
    print("-" * 60)
    print(f"Ranking improved: {improved}/{len(ranking_questions)}")
    print(f"Ranking unchanged: {unchanged}/{len(ranking_questions)}")
    print(f"Ranking regressed: {regressed}/{len(ranking_questions)}")
    print(f"Questions with changed top-three order: {order_changes}/{len(questions)}")
    print(
        "Average reranking time: "
        f"{sum(reranking_times.values()) / len(reranking_times):.4f} seconds"
    )

    print(
        "\nInterpretation: MRR measures the rank of the first labeled "
        "relevant section. It does not measure generated-answer correctness. "
        "Results depend on the quality and coverage of the relevance labels."
    )
    print(
        "Note: MIN_SCORE=-1.0 allows broad candidate retrieval. "
        "Unanswerable-question results must not be interpreted as a "
        "production answerability decision."
    )


if __name__ == "__main__":
    main()
