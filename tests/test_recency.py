"""Tests for date extraction and newest-first ordering."""

from datetime import date

from bankdoc.retrieval.recency import publication_date, sort_newest_first
from bankdoc.retrieval.retriever import RetrievedChunk


def make(uri: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=0, title="T", source_uri=uri, page_number=1, content="c", score=0.8)


def test_extracts_date_from_file_name():
    assert publication_date("mpr-2026-10-28.pdf") == date(2026, 10, 28)


def test_no_date_returns_none():
    assert publication_date("swp2026-39.pdf") is None


def test_invalid_date_returns_none():
    assert publication_date("mpr-2026-13-45.pdf") is None


def test_sorts_newest_first_and_undated_last():
    chunks = [make("mpr-2025-10-29.pdf"), make("swp2026-39.pdf"), make("mpr-2026-10-28.pdf")]
    ordered = [c.source_uri for c in sort_newest_first(chunks)]
    assert ordered == ["mpr-2026-10-28.pdf", "mpr-2025-10-29.pdf", "swp2026-39.pdf"]


def test_same_date_keeps_relevance_order():
    a, b = make("mpr-2026-10-28.pdf"), make("mpr-2026-10-28.pdf")
    a.chunk_id, b.chunk_id = 1, 2
    assert [c.chunk_id for c in sort_newest_first([a, b])] == [1, 2]
