"""Deterministic wording guard for model-year recall search results.

This application currently searches recalls by model year, make, and model.
It does NOT perform VIN-specific applicability or repair-status lookups.
"""
import re

VIN_NOTICE = (
    "**Recall status:** These records were retrieved by year, make, and model only. "
    "Whether a recall applies to this vehicle or remains unrepaired is **unverified**. "
    "Check the VIN at https://www.nhtsa.gov/recalls or with the manufacturer."
)

# Narrowly targeted to positive, vehicle-specific assertions; preserve advice such as
# "check for open recalls" and "don't assume a recall is open".
_REWRITES = (
    (r"\b(the|this|that|an?) open recall\b",
     r"\1 model-year-listed recall (VIN status unverified)"),
    (r"\b(the|these|those) open recalls\b",
     r"\1 model-year-listed recalls (VIN status unverified)"),
    (r"\b(the|this|that|an?) unrepaired recall\b",
     r"\1 model-year-listed recall (repair status unverified)"),
    (r"\b(the|this|that|an?) active recall\b",
     r"\1 model-year-listed recall (VIN status unverified)"),
    (r"\b(the|this|that) recall (?:is|remains) (?:still )?open\b",
     r"\1 recall's VIN-level repair status is unverified"),
)


def ensure_unverified_recall_status(answer: str, has_model_year_recalls: bool) -> str:
    """Remove unsupported positive status claims and add explicit VIN caveat.

    This is a secondary guardrail, not a substitute for prompt instructions or
    checking whether recall evidence is substantively relevant.
    """
    if not has_model_year_recalls:
        return answer

    safe = answer
    for pattern, replacement in _REWRITES:
        safe = re.sub(pattern, replacement, safe, flags=re.IGNORECASE)

    if "**Recall status:**" not in safe:
        safe = f"{safe.rstrip()}\n\n{VIN_NOTICE}"
    return safe
