"""Tests for answerability retrieval evaluation."""

import pytest

from app.answerability_evaluator import evaluate_answerability_retrieval


def test_evaluates_answerable_and_unanswerable_questions():
    question_results = [
        {
            "answerable": True,
            "expected_documents": ["crashloopbackoff.md"],
            "retrieved_documents": ["crashloopbackoff.md"],
        },
        {
            "answerable": True,
            "expected_documents": ["oomkilled.md"],
            "retrieved_documents": ["database-connectivity.md"],
        },
        {
            "answerable": False,
            "expected_documents": [],
            "retrieved_documents": [],
        },
        {
            "answerable": False,
            "expected_documents": [],
            "retrieved_documents": ["crashloopbackoff.md"],
        },
    ]

    result = evaluate_answerability_retrieval(question_results)

    assert result["total_questions"] == 4
    assert result["answerable_questions"] == 2
    assert result["unanswerable_questions"] == 2
    assert result["answerable_retrieval_success_rate"] == 0.5
    assert result["unanswerable_document_retrieval_rate"] == 0.5


def test_empty_results_return_none_rates():
    result = evaluate_answerability_retrieval([])

    assert result == {
        "total_questions": 0,
        "answerable_questions": 0,
        "unanswerable_questions": 0,
        "answerable_retrieval_success_rate": None,
        "unanswerable_document_retrieval_rate": None,
    }


def test_multiple_expected_documents_succeed_if_one_is_retrieved():
    question_results = [
        {
            "answerable": True,
            "expected_documents": [
                "crashloopbackoff.md",
                "database-connectivity.md",
            ],
            "retrieved_documents": ["database-connectivity.md"],
        }
    ]

    result = evaluate_answerability_retrieval(question_results)

    assert result["answerable_retrieval_success_rate"] == 1.0


@pytest.mark.parametrize(
    "invalid_results",
    [
        None,
        {},
        "invalid",
    ],
)
def test_rejects_invalid_input_container(invalid_results):
    with pytest.raises(TypeError):
        evaluate_answerability_retrieval(invalid_results)


def test_rejects_non_dictionary_question_result():
    with pytest.raises(TypeError, match="index 0"):
        evaluate_answerability_retrieval([None])


@pytest.mark.parametrize(
    "invalid_answerable",
    [
        "true",
        1,
        None,
    ],
)
def test_rejects_invalid_answerable_value(invalid_answerable):
    question_results = [
        {
            "answerable": invalid_answerable,
            "expected_documents": [],
            "retrieved_documents": [],
        }
    ]

    with pytest.raises(ValueError, match="answerable"):
        evaluate_answerability_retrieval(question_results)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("expected_documents", None),
        ("expected_documents", "crashloopbackoff.md"),
        ("retrieved_documents", None),
        ("retrieved_documents", "crashloopbackoff.md"),
    ],
)
def test_rejects_invalid_document_lists(field, invalid_value):
    question_result = {
        "answerable": True,
        "expected_documents": [],
        "retrieved_documents": [],
    }
    question_result[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        evaluate_answerability_retrieval([question_result])