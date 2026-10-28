"""Conservative, versioned weak-label taxonomy for NHTSA complaint components.

This file does NOT modify ml.prepare_nhtsa.COMPONENT_MAP or the existing
single-label / multilabel_v2 experiments. It fixes the exact-match alias
ENGINE AND ENGINE COOLING and uses more accurate names for POWER TRAIN.

Reported components are not confirmed fault diagnoses.
"""
from __future__ import annotations
import re

# Match component tokens exactly, or their colon-qualified subcomponents.
# Avoid substring matches such as ENGINE -> ENGINEERING.
COMPONENT_GROUPS_V3 = (
    (("AIR BAGS", "SEAT BELTS"), "Airbags / Safety"),
    (("SERVICE BRAKES", "PARKING BRAKE"), "Brakes"),
    (("ELECTRICAL SYSTEM", "EXTERIOR LIGHTING"), "Electrical"),
    (("ENGINE", "ENGINE AND ENGINE COOLING", "FUEL/PROPULSION SYSTEM", "VEHICLE SPEED CONTROL"), "Engine / Fuel"),
    (("STEERING", "SUSPENSION"), "Steering / Suspension"),
    (("POWER TRAIN",), "Powertrain / Drivetrain"),
)
LABELS_V3 = sorted({group for _, group in COMPONENT_GROUPS_V3})
LEGACY_PRIMARY_NAME_TO_V3 = {
    "Airbags / Safety": "Airbags / Safety",
    "Brakes": "Brakes",
    "Electrical": "Electrical",
    "Engine / Powertrain": "Engine / Fuel",
    "Steering / Suspension": "Steering / Suspension",
    "Transmission": "Powertrain / Drivetrain",
}


def components_to_labels_v3(raw: str) -> list[str]:
    """Map all applicable comma-separated NHTSA component names to groups."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    parts = [re.sub(r"\s+", " ", piece.strip().upper()) for piece in raw.split(",")]
    matches = set()
    for part in parts:
        for tokens, group in COMPONENT_GROUPS_V3:
            if any(part == token or part.startswith(token + ":") or part.startswith(token + " :")
                   for token in tokens):
                matches.add(group)
    return sorted(matches)


def normalize_summary(text: str) -> str:
    """Conservative normalization for duplicate text detection before splitting."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", str(text).casefold())).strip()
