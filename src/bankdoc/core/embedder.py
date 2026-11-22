"""Text embedding using a local sentence-transformers model.

An embedding turns a piece of text into a fixed-length vector of numbers.
Texts with similar meaning end up close together in vector space, which is
what makes semantic search possible.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from bankdoc.core.config import get_settings

# The bge model family is trained to expect this prefix on search queries
# (but NOT on the documents being indexed). It improves retrieval quality.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _load_model() -> SentenceTransformer:
    """Load the model once and cache it.

    Loading takes a few seconds and the first run downloads the weights
    (~130 MB) from Hugging Face, so we never want to repeat it.
    """
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed a batch of document chunks for indexing.

    Args:
        texts: The chunk texts to embed.

    Returns:
        A float32 array of shape (len(texts), embedding_dim). Vectors are
        L2-normalised, so cosine similarity equals the dot product.
    """
    model = _load_model()
    return model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embed_query(query: str) -> np.ndarray:
    """Embed a single user question for searching.

    Args:
        query: The raw user question.

    Returns:
        A float32 array of shape (embedding_dim,).
    """
    model = _load_model()
    return model.encode(
        QUERY_PREFIX + query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
