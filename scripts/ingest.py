"""Command-line entry point: ingest every PDF under a folder.

Access level is derived from the sub-folder name:
    data/raw/internal/...    -> 1
    data/raw/restricted/...  -> 2
    anything else            -> 0 (public)

Usage (from the project root):
    python scripts/ingest.py
    python scripts/ingest.py --dir data/raw --force
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bankdoc.core.db import close_pool
from bankdoc.ingestion.pipeline import ingest_file

ACCESS_LEVEL_BY_FOLDER = {"public": 0, "internal": 1, "restricted": 2}


def access_level_for(path: Path, root: Path) -> int:
    """Look at the first folder below root and map it to an access level."""
    relative_parts = path.relative_to(root).parts
    if len(relative_parts) > 1:
        return ACCESS_LEVEL_BY_FOLDER.get(relative_parts[0].lower(), 0)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs into the vector database.")
    parser.add_argument("--dir", default="data/raw", help="Folder containing PDF files")
    parser.add_argument("--force", action="store_true", help="Re-ingest files already stored")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(args.dir).resolve()
    pdf_files = sorted(root.rglob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found under {root}")
        return

    results = []
    try:
        for pdf in pdf_files:
            source_uri = pdf.relative_to(root).as_posix()
            level = access_level_for(pdf, root)
            results.append(ingest_file(pdf, source_uri, level, args.force))
    finally:
        close_pool()

    print("\n=== Ingestion summary ===")
    for r in results:
        print(f"{r.status:9} pages={r.pages:4} chunks={r.chunks:5}  {r.source_uri}")


if __name__ == "__main__":
    main()
