"""Ingestion pipeline: PDF -> pages -> chunks -> embeddings -> database."""

import logging
from dataclasses import dataclass
from pathlib import Path

from bankdoc.core.db import get_connection
from bankdoc.core.embedder import embed_documents
from bankdoc.ingestion import store
from bankdoc.ingestion.chunker import chunk_pages
from bankdoc.ingestion.loader import load_pdf

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Outcome of ingesting one file, used for the summary report."""

    source_uri: str
    status: str  # "ingested", "skipped" (already present) or "empty" (no text)
    pages: int = 0
    chunks: int = 0


def ingest_file(
    path: Path, source_uri: str, access_level: int = 0, force: bool = False
) -> IngestResult:
    """Ingest one PDF file.

    Slow work (parsing and embedding) happens BEFORE the database
    transaction opens, so the transaction stays short.

    Args:
        path: Location of the PDF on disk.
        source_uri: Stable identifier stored in the DB (relative path).
            It is the idempotency key: the same URI is never stored twice.
        access_level: 0 = public, 1 = internal, 2 = restricted.
        force: If True, delete the existing copy and ingest again.

    Returns:
        An IngestResult describing what happened.
    """
    # Cheap check first: skip before doing any expensive work.
    with get_connection() as conn:
        existing_id = store.find_document_id(conn, source_uri)
    if existing_id is not None and not force:
        logger.info("Skipping %s (already ingested)", source_uri)
        return IngestResult(source_uri, "skipped")

    document = load_pdf(path)
    chunks = chunk_pages(document.pages)
    if not chunks:
        logger.warning("No extractable text in %s (scanned PDF?)", source_uri)
        return IngestResult(source_uri, "empty")

    embeddings = embed_documents([c.content for c in chunks])

    # One transaction: replace old copy (if any) and insert the new one.
    with get_connection() as conn:
        if existing_id is not None:
            store.delete_document(conn, existing_id)
        document_id = store.insert_document(conn, source_uri, document.title, access_level)
        store.insert_chunks(conn, document_id, access_level, chunks, embeddings)

    logger.info("Ingested %s: %d pages, %d chunks", source_uri, len(document.pages), len(chunks))
    return IngestResult(source_uri, "ingested", len(document.pages), len(chunks))
