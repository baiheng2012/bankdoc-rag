"""Tests for citation parsing and the answering logic.

A fake LLM and a fake retriever replace the real ones, so these tests run
in milliseconds and need no database, GPU or network.
"""

import pytest

from bankdoc.generation import answerer
from bankdoc.generation.citations import extract_citation_numbers, split_valid_invalid
from bankdoc.generation.prompt import REFUSAL_TEXT, build_user_prompt
from bankdoc.retrieval.retriever import RetrievedChunk


def make_chunk(i: int, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=i, title="T", source_uri=f"mpr-2026-0{i}-01.pdf",
        page_number=i, content=f"content {i}", score=score,
    )


class FakeLLM:
    """Returns a canned reply and records whether it was called."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def generate(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


def patch_retrieval(monkeypatch, chunks):
    """Make answer_question use our chunks instead of querying the database."""
    monkeypatch.setattr(answerer, "retrieve", lambda question, clearance: chunks)


def test_extract_citations_dedupes_and_keeps_order():
    assert extract_citation_numbers("A [2][1]. B [2]. C [3].") == [2, 1, 3]


def test_split_valid_invalid():
    assert split_valid_invalid([1, 3, 9], source_count=3) == ([1, 3], [9])


def test_prompt_numbers_sources_from_one():
    prompt = build_user_prompt("Q?", [make_chunk(1), make_chunk(2)])
    assert "[1] (file: mpr-2026-01-01.pdf, published: 2026-01-01, page 1)" in prompt
    assert "[2] (file: mpr-2026-02-01.pdf, published: 2026-02-01, page 2)" in prompt
    assert prompt.endswith("QUESTION: Q?")


def test_low_scores_refuse_without_calling_llm(monkeypatch):
    patch_retrieval(monkeypatch, [make_chunk(1, score=0.2)])
    llm = FakeLLM("should never be used [1]")
    result = answerer.answer_question("q", llm=llm)
    assert result.refused and result.refusal_reason == "no_relevant_sources"
    assert llm.calls == 0


def test_valid_answer_returns_cited_chunks(monkeypatch):
    patch_retrieval(monkeypatch, [make_chunk(1), make_chunk(2)])
    result = answerer.answer_question("q", llm=FakeLLM("Inflation eases [1]."))
    assert not result.refused
    assert [c.chunk_id for c in result.citations] == [2]


def test_answer_without_citation_is_withheld(monkeypatch):
    patch_retrieval(monkeypatch, [make_chunk(1)])
    result = answerer.answer_question("q", llm=FakeLLM("Inflation eases."))
    assert result.refused and result.refusal_reason == "no_valid_citation"


def test_hallucinated_citation_only_is_withheld(monkeypatch):
    patch_retrieval(monkeypatch, [make_chunk(1)])
    result = answerer.answer_question("q", llm=FakeLLM("Inflation eases [7]."))
    assert result.refused and result.invalid_citations == [7]


def test_model_refusal_is_detected(monkeypatch):
    patch_retrieval(monkeypatch, [make_chunk(1)])
    result = answerer.answer_question("q", llm=FakeLLM(REFUSAL_TEXT))
    assert result.refused and result.refusal_reason == "model_refused"
