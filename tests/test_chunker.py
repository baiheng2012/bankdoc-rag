"""Unit tests for the chunker and text cleaner. No database needed."""

from bankdoc.ingestion.chunker import chunk_pages, chunk_text
from bankdoc.ingestion.loader import PageText, clean_text


def test_clean_text_rejoins_hyphenated_words():
    assert clean_text("infla-\ntion is high") == "inflation is high"


def test_clean_text_merges_single_newlines_keeps_paragraphs():
    assert clean_text("line one\nline two\n\nnew paragraph") == "line one line two\n\nnew paragraph"


def test_short_text_is_one_chunk():
    assert chunk_text("One short sentence.") == ["One short sentence."]


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []


def test_chunks_respect_max_size():
    text = " ".join(f"This is sentence number {i}." for i in range(200))
    chunks = chunk_text(text, max_chars=300, overlap_chars=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_consecutive_chunks_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(100))
    chunks = chunk_text(text, max_chars=300, overlap_chars=80)
    # The last sentence of chunk N must reappear at the start of chunk N+1.
    last_sentence = chunks[0].split(". ")[-1]
    assert last_sentence[:20] in chunks[1]


def test_giant_sentence_is_hard_split():
    chunks = chunk_text("x" * 2500, max_chars=1000, overlap_chars=100)
    assert all(len(c) <= 1000 for c in chunks)
    assert sum(len(c) for c in chunks) >= 2500


def test_chunk_pages_keeps_page_numbers_and_sequential_index():
    pages = [
        PageText(1, "A sentence on page one that is long enough to keep as a chunk. " * 2),
        PageText(2, "A sentence on page two that is long enough to keep as a chunk. " * 2),
    ]
    chunks = chunk_pages(pages)
    assert [c.page_number for c in chunks] == [1, 2]
    assert [c.chunk_index for c in chunks] == [0, 1]


def test_tiny_noise_chunks_are_dropped():
    assert chunk_pages([PageText(1, "12")]) == []
