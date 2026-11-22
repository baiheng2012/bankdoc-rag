"""Question answering orchestration: retrieve -> generate -> verify."""

import logging
from dataclasses import dataclass, field

from bankdoc.core.config import get_settings
from bankdoc.generation.citations import extract_citation_numbers, split_valid_invalid
from bankdoc.generation.llm import LLMProvider, get_llm
from bankdoc.generation.prompt import REFUSAL_TEXT, SYSTEM_PROMPT, build_user_prompt
from bankdoc.retrieval.retriever import RetrievedChunk, retrieve
from bankdoc.retrieval.recency import sort_newest_first


logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    """Everything the caller (CLI, API, audit log, evaluator) needs to know."""

    answer: str
    refused: bool
    refusal_reason: str | None = None  # "no_relevant_sources", "model_refused", "no_valid_citation"
    citations: list[RetrievedChunk] = field(default_factory=list)  # chunks actually cited
    invalid_citations: list[int] = field(default_factory=list)  # hallucinated numbers
    retrieved: list[RetrievedChunk] = field(default_factory=list)  # everything retrieved


def _refuse(reason: str, retrieved: list[RetrievedChunk]) -> AnswerResult:
    """Build a standard refusal result."""
    return AnswerResult(answer=REFUSAL_TEXT, refused=True, refusal_reason=reason, retrieved=retrieved)


def answer_question(
    question: str, clearance: int = 0, llm: LLMProvider | None = None
) -> AnswerResult:
    """Answer a question using only the permitted documents, with verified citations.

    Args:
        question: The user's question.
        clearance: The user's access level, passed to the retriever.
        llm: LLM provider. Injected in tests; defaults to the configured one.

    Returns:
        An AnswerResult. The answer is returned only if it carries at least
        one valid citation; otherwise the standard refusal is returned.
    """
    settings = get_settings()
    retrieved = retrieve(question, clearance)

    # Guard 1: nothing relevant enough was found -> refuse WITHOUT calling the LLM.
    #relevant = [c for c in retrieved if c.score >= settings.min_score]
    # Keep only relevant chunks, then put the newest reports first so the
    # model sees (and cites) the most recent forecast before older ones.
    relevant = sort_newest_first([c for c in retrieved if c.score >= settings.min_score])
    if not relevant:
        logger.info("Refusing: no chunk above min_score=%.2f", settings.min_score)
        return _refuse("no_relevant_sources", retrieved)

    provider = llm or get_llm()
    raw_answer = provider.generate(SYSTEM_PROMPT, build_user_prompt(question, relevant)).strip()

    # The model itself said the sources do not contain the answer.
    if REFUSAL_TEXT.lower() in raw_answer.lower():
        return _refuse("model_refused", retrieved)

    # Guard 2: every accepted answer must cite real sources.
    numbers = extract_citation_numbers(raw_answer)
    valid, invalid = split_valid_invalid(numbers, len(relevant))
    if not valid:
        logger.warning("Answer withheld: no valid citation. Raw answer: %s", raw_answer)
        result = _refuse("no_valid_citation", retrieved)
        result.invalid_citations = invalid
        return result

    return AnswerResult(
        answer=raw_answer,
        refused=False,
        citations=[relevant[n - 1] for n in valid],
        invalid_citations=invalid,
        retrieved=retrieved,
    )
