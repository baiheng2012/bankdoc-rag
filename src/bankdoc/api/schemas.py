"""Request and response models.

These Pydantic classes define the public contract of the API. FastAPI uses
them to (1) validate incoming JSON, (2) filter outgoing JSON, and (3)
generate the interactive docs at /docs.
"""

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    """Body of POST /ask."""

    # extra="forbid": unknown fields are rejected with 422. In particular a
    # client cannot sneak in a "clearance" field and hope it is honoured.
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=1000)


class CitationOut(BaseModel):
    """One cited source, with enough detail for a human to verify the claim."""

    source_uri: str  # file name, contains the report date
    title: str
    page_number: int
    score: float  # retrieval similarity, 0..1
    snippet: str  # the start of the cited passage


class AskResponse(BaseModel):
    """Body returned by POST /ask."""

    answer: str
    refused: bool
    refusal_reason: str | None = None
    citations: list[CitationOut] = Field(default_factory=list)
