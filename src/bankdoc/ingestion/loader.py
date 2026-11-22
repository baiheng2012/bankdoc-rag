"""PDF loading and text cleaning.

Responsibility: turn a PDF file into a list of cleaned page texts. This
module knows nothing about chunking, embeddings or the database.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class PageText:
    """The cleaned text of one PDF page."""

    page_number: int  # 1-based, matches what a human sees in a PDF viewer
    text: str


@dataclass
class LoadedDocument:
    """A parsed PDF: its title and the non-empty pages."""

    title: str
    pages: list[PageText]


def clean_text(raw: str) -> str:
    """Normalise text extracted from a PDF.

    PDF extraction produces artefacts that hurt retrieval quality:
      - words split across lines with a hyphen ("infla-\\ntion")
      - hard line breaks in the middle of a sentence
      - runs of spaces and null characters

    Args:
        raw: Text exactly as returned by pypdf.

    Returns:
        Cleaned text. Blank lines (paragraph breaks) are preserved.
    """
    text = raw.replace("\x00", "")
    # Re-join words hyphenated at a line break: "infla-\ntion" -> "inflation".
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # A single newline inside a paragraph becomes a space.
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse runs of spaces/tabs.
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into one paragraph break.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> LoadedDocument:
    """Read a PDF file and return its title and cleaned, non-empty pages.

    Pages with no extractable text (for example scanned images) are skipped.
    OCR is out of scope for this project.

    Args:
        path: Path to the PDF file.

    Returns:
        A LoadedDocument. If the PDF has no title metadata, the file name
        (without extension) is used as the title.
    """
    reader = PdfReader(str(path))

    # Prefer the title stored in the PDF metadata, fall back to the file name.
    metadata_title = (reader.metadata.title or "").strip() if reader.metadata else ""
    title = metadata_title or path.stem

    pages: list[PageText] = []
    for index, page in enumerate(reader.pages, start=1):
        cleaned = clean_text(page.extract_text() or "")
        if cleaned:
            pages.append(PageText(page_number=index, text=cleaned))

    return LoadedDocument(title=title, pages=pages)
