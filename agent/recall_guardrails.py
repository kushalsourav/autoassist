"""Deterministic wording guard for model-year recall search results.

This application currently searches recalls by model year, make, and model.
It does NOT perform VIN-specific applicability or repair-status lookups.

Design
------
The guardrail has two layers:

1. **Phrase-level rewrites** (``_REWRITES``): narrow regex substitutions that
   turn false-certainty phrasing ("this open recall") into honest, qualified
   wording ("this model-year-listed recall (VIN status unverified)").
   Rules deliberately avoid touching general advice such as
   "check for open recalls" or "don't assume a recall is open/closed".

2. **VIN notice** (``VIN_NOTICE``): a trailing disclaimer appended whenever
   any model-year recall evidence was surfaced in the answer.  Skipped when
   the answer already contains an equivalent grounded caveat.

Improving recall retrieval signal
-----------------------------------
The retriever in ``agent/rag.py`` ranks NHTSA records by symptom similarity
only.  For better recall precision:

* Use both symptom *and* component-category signal — ``classification`` from
  the DistilBERT model is already propagated into ``retrieve()``; make sure
  ``RAG_BACKEND=faiss`` and ``RAG_CORPUS_MODE=prefer`` are set.
* Prefer ``RAG_MIN_SIMILARITY=0.40`` (up from 0.35) to cut low-confidence
  matches that are technically above the floor but mislead the LLM.
* The corpus builder (``scripts/build_recall_corpus_v2.py``) can be re-run
  after adding more NHTSA records to improve index coverage.
"""
from __future__ import annotations

import re

VIN_NOTICE = (
    "**Recall status:** These records were retrieved by year, make, and model only. "
    "Whether a recall applies to this vehicle or remains unrepaired is **unverified**. "
    "Check the VIN at https://www.nhtsa.gov/recalls or with the manufacturer."
)

# ──────────────────────────────────────────────────── phrase-level rewrites
#
# Each tuple is (pattern, replacement).  All patterns use re.IGNORECASE.
# Only positive, vehicle-specific assertions are targeted; general advice
# ("check for open recalls", "don't assume a recall is open") is left intact.
#
_REWRITES: tuple[tuple[str, str], ...] = (
    # "the open recall" / "this open recall" / "an open recall" / "that open recall"
    (
        r"\b(the|this|that|an?)\s+open\s+recall\b",
        r"\1 model-year-listed recall (VIN status unverified)",
    ),
    # "the open recalls" / "these open recalls" / "those open recalls"
    (
        r"\b(the|these|those)\s+open\s+recalls\b",
        r"\1 model-year-listed recalls (VIN status unverified)",
    ),
    # "the unrepaired recall" / "a/an unrepaired recall"
    (
        r"\b(the|this|that|an?)\s+unrepaired\s+recall\b",
        r"\1 model-year-listed recall (repair status unverified)",
    ),
    # "the active recall" / "an active recall"
    (
        r"\b(the|this|that|an?)\s+active\s+recall\b",
        r"\1 model-year-listed recall (VIN status unverified)",
    ),
    # "the recall is open" / "the recall remains open" / "the recall is still open"
    (
        r"\b(the|this|that)\s+recall\s+(?:is|remains)\s+(?:still\s+)?open\b",
        r"\1 recall's VIN-level repair status is unverified",
    ),
    # "recall is outstanding" / "recall remains outstanding"
    (
        r"\b(the|this|that)\s+recall\s+(?:is|remains)\s+outstanding\b",
        r"\1 recall's repair status is unverified (model-year scope only)",
    ),
    # "your vehicle has an open/active recall"
    (
        r"\byour\s+vehicle\s+has\s+(?:an?\s+)?(?:open|active|outstanding)\s+recall\b",
        r"your vehicle may be within the model-year scope of a recall (VIN status unverified)",
    ),
    # "there is an open recall" / "there is an active recall"
    (
        r"\bthere\s+(?:is|are)\s+(?:an?\s+)?(?:open|active|outstanding)\s+recall[s]?\b",
        r"there is a model-year-listed recall (VIN status unverified)",
    ),
    # "recall has not been repaired" / "recall repair is pending"
    (
        r"\brecall\s+(?:repair\s+)?(?:has|have)\s+not\s+been\s+(?:repaired|completed|performed)\b",
        r"recall repair status is unverified (model-year scope only)",
    ),
)


def ensure_unverified_recall_status(answer: str, has_model_year_recalls: bool) -> str:
    """Remove unsupported positive-status claims and append the VIN caveat.

    This is a **secondary** guardrail, not a substitute for the system-prompt
    instructions or the grounded answer check in ``agent/runtime/grounding.py``.

    Parameters
    ----------
    answer:
        The raw text produced by the language model.
    has_model_year_recalls:
        ``True`` when the NHTSA tool returned at least one model-year record
        *and* the retriever selected at least one evidence candidate.  When
        ``False`` the function is a no-op so that non-recall answers are never
        polluted with unsolicited caveats.

    Returns
    -------
    str
        The possibly-rewritten answer with the VIN notice appended (at most
        once) when applicable.
    """
    if not has_model_year_recalls:
        return answer

    safe = answer
    for pattern, replacement in _REWRITES:
        safe = re.sub(pattern, replacement, safe, flags=re.IGNORECASE)

    # Append the notice only if the answer does not already contain it.
    if "**Recall status:**" not in safe:
        safe = f"{safe.rstrip()}\n\n{VIN_NOTICE}"
    return safe
