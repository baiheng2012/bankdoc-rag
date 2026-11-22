"""FastAPI application: the HTTP layer.

This file contains NO business logic. It authenticates, calls
answer_question(), writes the audit record and shapes the response. All
real work happens in the generation/ and retrieval/ packages, which keeps
them testable without HTTP.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from bankdoc.api.auth import User, get_current_user
from bankdoc.api.schemas import AskRequest, AskResponse, CitationOut
from bankdoc.core.audit import write_audit_log
from bankdoc.core.db import close_pool
from bankdoc.core.embedder import embed_query
from bankdoc.generation.answerer import answer_question

logger = logging.getLogger(__name__)

SNIPPET_CHARS = 300


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks.

    Code before `yield` runs once at startup; code after it runs at
    shutdown. We load the embedding model at startup so the first real
    request is not slow, and close the DB pool cleanly on shutdown.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    embed_query("warm up")
    logger.info("Embedding model loaded; service ready")
    yield
    close_pool()


app = FastAPI(title="bankdoc-rag", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness probe for Docker / Kubernetes. Needs no authentication."""
    return {"status": "ok"}


@app.get("/me")
def me(user: User = Depends(get_current_user)) -> dict:
    """Return the identity the server derived from the API key (handy for debugging)."""
    return {"user_id": user.user_id, "clearance": user.clearance}


# Note: a plain `def`, not `async def`. answer_question() blocks (database and
# LLM calls). FastAPI runs plain `def` endpoints in a worker thread pool, so a
# slow request does not freeze the whole server. With `async def` it would.
@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, user: User = Depends(get_current_user)) -> AskResponse:
    """Answer a question with citations, respecting the caller's clearance."""
    # The clearance comes from the authenticated user, never from the request body.
    result = answer_question(body.question, clearance=user.clearance)

    # Audit EVERY request, including refusals. If this raises, the request
    # fails with 500 and the answer is not returned (fail closed).
    write_audit_log(
        user_id=user.user_id,
        question=body.question,
        answer=result.answer,
        cited_chunk_ids=[c.chunk_id for c in result.citations],
        blocked=result.refused,
        block_reason=result.refusal_reason,
    )

    return AskResponse(
        answer=result.answer,
        refused=result.refused,
        refusal_reason=result.refusal_reason,
        citations=[
            CitationOut(
                source_uri=c.source_uri,
                title=c.title,
                page_number=c.page_number,
                score=round(c.score, 3),
                snippet=c.content[:SNIPPET_CHARS],
            )
            for c in result.citations
        ],
    )
