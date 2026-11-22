"""Question answering orchestration: guard -> retrieve -> generate -> verify.

Pipeline and the guardrail applied at each stage:
  1. redact PII in the question          (guardrails/pii.py)
  2. block prompt-injection questions    (guardrails/injection.py)
  3. retrieve, keep relevant, newest first
  4. drop chunks that contain instructions (indirect injection)
  5. generate with the LLM
  6. require valid citations, block system-prompt leaks,
     require numbers to be grounded in the cited chunks, redact output PII
"""

import logging
from dataclasses import dataclass, field

from bankdoc.core.config import get_settings
from bankdoc.generation.citations import extract_citation_numbers, split_valid_invalid
from bankdoc.generation.llm import LLMProvider, get_llm
from bankdoc.generation.prompt import REFUSAL_TEXT, SYSTEM_PROMPT, build_user_prompt
from bankdoc.guardrails.injection import detect_injection
from bankdoc.guardrails.output_checks import find_ungrounded_numbers, leaks_system_prompt
from bankdoc.guardrails.pii import redact_pii
from bankdoc.retrieval.recency import sort_newest_first
from bankdoc.retrieval.retriever import RetrievedChunk, retrieve
from bankdoc.guardrails.language import is_supported_script
from bankdoc.guardrails.pii import redact_pii, strip_placeholders



logger = logging.getLogger(__name__)

BLOCKED_TEXT = "Your request was blocked by a security policy."
UNVERIFIED_TEXT = "The generated answer could not be verified against the sources, so it was withheld."
UNSUPPORTED_LANGUAGE_TEXT = "Please ask your question in English; other scripts are not supported yet."

@dataclass
class AnswerResult:
    """Everything the caller (CLI, API, audit log, evaluator) needs to know."""

    answer: str
    refused: bool
    # Reasons: no_relevant_sources, model_refused, no_valid_citation, prompt_injection,
    #          unsafe_sources, output_leak, ungrounded_numbers
    #unsupported_language
    refusal_reason: str | None = None
    citations: list[RetrievedChunk] = field(default_factory=list)  # chunks actually cited
    invalid_citations: list[int] = field(default_factory=list)  # hallucinated numbers
    retrieved: list[RetrievedChunk] = field(default_factory=list)  # everything retrieved
    # The question AFTER PII redaction. This is the only version that may be
    # logged or stored.
    redacted_question: str = ""
    pii_findings: dict[str, int] = field(default_factory=dict)  # e.g. {"EMAIL": 1}
    ungrounded_numbers: list[str] = field(default_factory=list)


def _refuse(
    reason: str, retrieved: list[RetrievedChunk], text: str = REFUSAL_TEXT
) -> AnswerResult:
    """Build a standard refusal result."""
    return AnswerResult(answer=text, refused=True, refusal_reason=reason, retrieved=retrieved)


def answer_question(
    question: str, clearance: int = 0, llm: LLMProvider | None = None
) -> AnswerResult:
    """Answer a question using only the permitted documents, with verified citations.

    Args:
        question: The user's raw question (may contain PII).
        clearance: The user's access level, passed to the retriever.
        llm: LLM provider. Injected in tests; defaults to the configured one.

    Returns:
        An AnswerResult. An answer is returned only if it passes every guardrail.
    """
    settings = get_settings()

    # Guardrail 1: redact PII. Everything below uses safe_question, never the raw one.
    redaction = redact_pii(question)
    safe_question = redaction.text
    if redaction.findings:
        # Log only the TYPES and COUNTS. Logs must never contain the values.
        logger.info("PII redacted from question: %s", redaction.findings)

    def finish(result: AnswerResult) -> AnswerResult:
        """Attach the redacted question and PII summary to any result."""
        result.redacted_question = safe_question
        result.pii_findings = redaction.findings
        return result

    # Guardrail 2: stop obvious injection attacks before spending any compute.
    # Guardrail 1b: script check. Chinese, Arabic, Cyrillic ... cannot be
    # searched reliably or inspected by the English injection rules.
    if not is_supported_script(safe_question):
        logger.info("Refusing: unsupported script")
        return finish(_refuse("unsupported_language", [], UNSUPPORTED_LANGUAGE_TEXT))

    # (existing code) Guardrail 2: injection check on safe_question stays here.    
    
    matched_rules = detect_injection(safe_question)
    if matched_rules:
        logger.warning("Blocked question: prompt-injection rules %s", matched_rules)
        return finish(_refuse("prompt_injection", [], BLOCKED_TEXT))
    search_query = strip_placeholders(safe_question)
    # Refuse only if redaction removed (almost) everything, i.e. the question
    # was essentially nothing but PII. `redaction.findings` is non-empty only
    # when something was actually redacted, so a short question that
    # contained no PII is NOT affected by this check.
    if redaction.findings and len(search_query) < 3:
        return finish(_refuse("no_relevant_sources", []))
    #retrieved = retrieve(safe_question, clearance)
    
    retrieved = retrieve(search_query, clearance)
    # Keep only relevant chunks, newest report first.
    candidates = sort_newest_first([c for c in retrieved if c.score >= settings.min_score])

    # Guardrail 3: indirect injection. A chunk that contains instructions is
    # dropped; it is data and must never be allowed to act as a command.
    relevant: list[RetrievedChunk] = []
    for chunk in candidates:
        chunk_rules = detect_injection(chunk.content)
        if chunk_rules:
            logger.warning("Dropped chunk %d: instruction-like text %s", chunk.chunk_id, chunk_rules)
            continue
        relevant.append(chunk)

    if not relevant:
        reason = "unsafe_sources" if candidates else "no_relevant_sources"
        logger.info("Refusing: %s", reason)
        return finish(_refuse(reason, retrieved))

    provider = llm or get_llm()
    raw_answer = provider.generate(SYSTEM_PROMPT, build_user_prompt(safe_question, relevant)).strip()

    # The model itself said the sources do not contain the answer.
    if REFUSAL_TEXT.lower() in raw_answer.lower():
        return finish(_refuse("model_refused", retrieved))

    # Guardrail 4: every accepted answer must cite real sources.
    numbers = extract_citation_numbers(raw_answer)
    valid, invalid = split_valid_invalid(numbers, len(relevant))
    if not valid:
        logger.warning("Answer withheld: no valid citation")
        result = _refuse("no_valid_citation", retrieved)
        result.invalid_citations = invalid
        return finish(result)

    # Guardrail 5: the answer must not reproduce our system prompt.
    if leaks_system_prompt(raw_answer, SYSTEM_PROMPT):
        logger.warning("Answer withheld: system prompt leakage")
        return finish(_refuse("output_leak", retrieved, BLOCKED_TEXT))

    # Guardrail 6: every number in the answer must appear in a cited chunk.
    cited = [relevant[n - 1] for n in valid]
    ungrounded = find_ungrounded_numbers(raw_answer, cited)
    if ungrounded:
        logger.warning("Answer withheld: ungrounded numbers %s", ungrounded)
        result = _refuse("ungrounded_numbers", retrieved, UNVERIFIED_TEXT)
        result.ungrounded_numbers = ungrounded
        return finish(result)

    # Guardrail 7: redact PII that may have leaked into the answer text.
    return finish(
        AnswerResult(
            answer=redact_pii(raw_answer).text,
            refused=False,
            citations=cited,
            invalid_citations=invalid,
            retrieved=retrieved,
        )
    )
