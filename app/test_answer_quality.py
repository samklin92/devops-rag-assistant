"""Tests for end-to-end RAG answer-quality metrics."""

import pytest

from app.answer_quality import summarize_answer_quality


def test_empty_results_return_empty_summary():
    result = summarize_answer_quality([])

    assert result == {
        "total_questions": 0,
        "relevance_evaluated": 0,
        "relevance_rate": None,
        "grounding_evaluated": 0,
        "grounding_rate": None,
        "answerability_evaluated": 0,
        "answerability_handling_evaluated": 0,
        "answerability_handling_rate": None,
    }


def test_summary_calculates_quality_rates():
    results = [
        {
            "answer_relevant": True,
            "grounded": True,
            "answerable": True,
            "answerability_handled": True,
        },
        {
            "answer_relevant": True,
            "grounded": False,
            "answerable": True,
            "answerability_handled": True,
        },
        {
            "answer_relevant": False,
            "grounded": True,
            "answerable": False,
            "answerability_handled": False,
        },
        {
            "answer_relevant": None,
            "grounded": None,
            "answerable": False,
            "answerability_handled": True,
        },
    ]

    result = summarize_answer_quality(results)

    assert result["total_questions"] == 4
    assert result["relevance_evaluated"] == 3
    assert result["relevance_rate"] == pytest.approx(2 / 3)
    assert result["grounding_evaluated"] == 3
    assert result["grounding_rate"] == pytest.approx(2 / 3)
    assert result["answerability_evaluated"] == 4
    assert result["answerability_handling_evaluated"] == 4
    assert result["answerability_handling_rate"] == pytest.approx(3 / 4)


def test_missing_assessments_are_not_counted_as_failures():
    results = [
        {"answer_relevant": True, "grounded": None},
        {"answer_relevant": None, "grounded": False},
    ]

    result = summarize_answer_quality(results)

    assert result["relevance_evaluated"] == 1
    assert result["relevance_rate"] == 1.0
    assert result["grounding_evaluated"] == 1
    assert result["grounding_rate"] == 0.0
    assert result["answerability_evaluated"] == 0
    assert result["answerability_handling_evaluated"] == 0
    assert result["answerability_handling_rate"] is None


@pytest.mark.parametrize(
    "invalid_results",
    [
        None,
        {},
        "not a list",
    ],
)
def test_invalid_results_container_is_rejected(invalid_results):
    with pytest.raises((TypeError, ValueError)):
        summarize_answer_quality(invalid_results)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("answer_relevant", "yes"),
        ("grounded", 1),
        ("answerable", 0),
        ("answerability_handled", "false"),
    ],
)
def test_invalid_assessment_values_are_rejected(field, invalid_value):
    results = [{field: invalid_value}]

    with pytest.raises(TypeError, match=field):
        summarize_answer_quality(results)


def test_non_dictionary_result_is_rejected():
    with pytest.raises(TypeError, match="index 0"):
        summarize_answer_quality([None])