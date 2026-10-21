from app import evaluator


def test_recall_counts_unique_documents():
    question = {
        "id": "q-test",
        "question": "Why was my container killed?",
        "expected_documents": [
            "crashloopbackoff.md",
            "oomkilled.md",
        ],
        "answerable": True,
    }

    retrieved_chunks = [
        {"source": "crashloopbackoff.md"},
        {"source": "crashloopbackoff.md"},
        {"source": "database-connectivity.md"},
    ]

    result = evaluator.evaluate_retrieval(
        question,
        retrieved_chunks,
    )

    assert result["recall"] == 0.5
    assert result["matched_documents"] == ["crashloopbackoff.md"]
    assert result["retrieved_documents"] == [
        "crashloopbackoff.md",
        "database-connectivity.md",
    ]


def test_unanswerable_question_has_no_recall():
    question = {
        "id": "q-test",
        "question": "How should I price a SaaS subscription?",
        "expected_documents": [],
        "answerable": False,
    }

    result = evaluator.evaluate_retrieval(
        question,
        [],
    )

    assert result["recall"] is None
    assert result["matched_documents"] == []


def test_missing_expected_documents_are_reported(
    monkeypatch,
    tmp_path,
):
    question = {
        "id": "q-test",
        "question": "What does exit code 137 mean?",
        "expected_documents": [
            "crashloopbackoff.md",
            "oomkilled.md",
        ],
        "answerable": True,
    }

    # Create a temporary knowledge base containing only one document.
    (tmp_path / "crashloopbackoff.md").write_text(
        "# CrashLoopBackOff",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluator,
        "KNOWLEDGE_BASE",
        tmp_path,
    )

    result = evaluator.evaluate_retrieval(
        question,
        [{"source": "crashloopbackoff.md"}],
    )

    assert result["missing_documents"] == ["oomkilled.md"]

def test_summarize_evaluation_calculates_metrics():
    results = [
        {
            "answerable": True,
            "recall": 1.0,
            "retrieved_documents": ["crashloopbackoff.md"],
        },
        {
            "answerable": True,
            "recall": 0.5,
            "retrieved_documents": ["oomkilled.md"],
        },
        {
            "answerable": False,
            "recall": None,
            "retrieved_documents": [],
        },
        {
            "answerable": False,
            "recall": None,
            "retrieved_documents": ["database-connectivity.md"],
        },
    ]

    summary = evaluator.summarize_evaluation(results)

    assert summary["total_questions"] == 4
    assert summary["answerable_questions"] == 2
    assert summary["unanswerable_questions"] == 2
    assert summary["mean_recall"] == 0.75
    assert summary["unanswerable_document_retrieval_rate"] == 0.5


def test_summarize_evaluation_handles_empty_results():
    summary = evaluator.summarize_evaluation([])

    assert summary["total_questions"] == 0
    assert summary["answerable_questions"] == 0
    assert summary["unanswerable_questions"] == 0
    assert summary["mean_recall"] is None
    assert summary["unanswerable_document_retrieval_rate"] is None