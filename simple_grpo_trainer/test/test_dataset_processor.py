import pytest
from simple_grpo_trainer.src.dataset_processor import extract_answer


def test_extract_answer_standard():
    example = {"answer": "Some context. ### The answer is 42."}
    result = extract_answer(example.copy())
    assert result["answer"] == "The answer is 42."


def test_extract_answer_with_commas():
    example = {"answer": "Intro. ### 1,234,567 is the answer."}
    result = extract_answer(example.copy())
    assert result["answer"] == "1234567 is the answer."


def test_extract_answer_no_marker():
    example = {"answer": "No marker here."}
    with pytest.raises(ValueError):
        extract_answer(example.copy())


def test_extract_answer_only_marker():
    example = {"answer": "### "}
    result = extract_answer(example.copy())
    assert result["answer"] == ""


def test_extract_answer_multiple_markers():
    example = {"answer": "foo ### first ### second"}
    result = extract_answer(example.copy())
    assert result["answer"] == "first ### second"
