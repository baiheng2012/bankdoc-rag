# ADR 0003: Numeric grounding checks against every source sent to the model

Status: accepted

## Context
Central-bank answers must not contain invented figures. The guardrail requires every number in an answer to appear in the supplied sources. First version: only in the chunks the answer cites.

## Evidence
- Blocked correctly: for "GDP growth in 2026 in the October report" the 7B model computed 1.2 + 0.3 = 1.5 itself. The figure 1.5 exists in no source, and the true value is 1%. The answer was withheld.
- Blocked wrongly: for the import-tariff question the model wrote 0.26, a real figure, but cited sources [2][4]; the figure was in source [7]. The answer was withheld for a citation-precision error, not a fabrication.

## Decision
Check numbers against all chunks sent to the model, not only the cited ones. Both cases are pinned by unit tests (a number found in an uncited source passes; a number found in no source is blocked).

## Consequences
- Fabricated and self-computed numbers are still blocked.
- Not detected: a real figure attached to the wrong report, or a correct figure answering a different question. Citation precision and faithfulness need an LLM judge (roadmap).
- Numeric grounding is a strict string match, so it can false-block formatting differences. The evaluation reports how often it fires.
