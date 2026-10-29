"""Conservative, inspectable recall *evidence* gate for OFFLINE evaluation.

A campaign ID and year/make/model membership do not establish VIN eligibility,
open-recall status, diagnosis, or causal relation to a complaint.

No ML labels or gold judgments are read by this module. A returned `mechanism`
means a plausible *component* connection, NOT an explicit documented symptom.
No model score is treated as a calibrated probability.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class EvidenceDecision:
    tier: str                     # direct | mechanism | abstain
    matched_symptoms: tuple[str, ...]
    supported_by: tuple[str, ...] # exact snippets from recall text
    reason: str


# Expression patterns are automotive-domain *development heuristics*. They are
# intentionally narrow to avoid conflating e.g. a tailgate and passenger door,
# a dashboard warning label and a live warning indicator, or a camera and SRS.
# Review a new independently labeled sample before using in production.
QUERY_SIGNALS = {
    'wiper_failure': r'\bwipers?\b',
    'camera_failure': r'\b(?:rear\s?view|back\s?up|reverse)\s+camera\b|\bcamera\b',
    'seatbelt_latch': r'\b(?:seat\s?belt|belt)\b.{0,55}\b(?:buckl\w*|latch\w*|lock\w*)\b',
    'airbag_warning': r'\b(?:air\s?bag|srs)\b.{0,60}\b(?:warning|indicator|light|lamp|sensor)\b',
    'engine_stall': r'\b(?:stall\w*|hesitat\w*|los(?:e|es|ing|t)\s+(?:engine\s+)?power|power\s+loss|shut\s*off|engine\s+(?:cuts?\s*out|dies?))\b',
    'brake_assist': r'\b(?:brak\w*|stopping\s+distance)\b.{0,90}\b(?:pedal|assist|stop|soft|hard|distance|fail|weak)\b|\b(?:hard|soft)\s+brake\s+pedal\b',
    'gear_shift': r'\b(?:gear|transmission|shifts?|propulsion)\b.{0,80}\b(?:shift|lose|los(?:es|ing|t)|unintended|unexpected|neutral|gear|propulsion)\b',
    'door_unlatch': r'\b(?:passenger|driver|front|rear|car|truck|the)\s+door\b|\bdoors?\b',
    'steering_heavy': r'\bsteer\w*\b.{0,70}\b(?:heavy|difficult|hard|assist|turning)\b',
    'steering_vibration': r'\b(?:steer\w*|wheel)\b.{0,80}\b(?:shak\w*|vibrat\w*|wobbl\w*)\b',
    'light_flicker': r'\b(?:lights?|lamps?|headlights?|dashboard)\b.{0,70}\b(?:flicker\w*|flash\w*|dim\w*|blinking)\b',
    'ac_noise': r'\b(?:air\s+conditioning|a\s?/?c|hvac)\b.{0,90}\b(?:nois\w*|squeak\w*|rattl\w*)\b|\b(?:squeak\w*|rattl\w*)\b.{0,90}\b(?:air\s+conditioning|a\s?/?c|hvac)\b',
    'audio_disconnect': r'\b(?:bluetooth|audio|infotainment)\b.{0,60}\b(?:disconnect\w*|cut\w*\s*out|fail\w*)\b',
}
# For a DIRECT match, the source text must itself describe the relevant failure
# mode. Generic categorization is not enough. Matching snippets are extracted
# from text, not synthesized as quotations.
DIRECT_PATTERNS = {
    'wiper_failure': r'\bwipers?\b.{0,100}\b(?:malfunction\w*|fail\w*|stop\w*|disrupt\w*)\b|\b(?:malfunction\w*|fail\w*|disrupt\w*)\b.{0,100}\bwipers?\b',
    'camera_failure': r'\b(?:camera|back.up camera|rearview camera)\b.{0,90}\b(?:malfunction\w*|fail\w*|not\s+function\w*|black\s+screen|display\s+may\s+not\s+function)\b|\b(?:malfunction\w*|fail\w*|disrupt\w*)\b.{0,110}\b(?:rearview\s+)?camera\b',
    'seatbelt_latch': r'\b(?:seat\s?belt|buckle)\b.{0,130}\b(?:prevent\w*.*?latch\w*|not\s+latch\w*|unlatch\w*|latching)\b',
    'airbag_warning': r'\b(?:air\s?bag|srs)\b.{0,90}\b(?:indicator|warning\s+light|lamp\s+illuminat\w*|warning\s+illuminat\w*)\b',
    'engine_stall': r'\b(?:engine|vehicle)\b.{0,130}\b(?:stall\w*|shut\w*\s*off|lose\w*\s+(?:engine\s+)?power)\b|\bstall\w*\b.{0,60}\bengine\b',
    'brake_assist': r'\b(?:brak\w*|vacuum\s+pump)\b.{0,120}\b(?:loss\s+of\s+(?:brak\w*|power\s+brak\w*)\s+assist|decreas\w*\s+(?:the\s+amount\s+of\s+)?vacuum|brak\w*\s+assist\s+fail\w*|stopping\s+distance\s+increas\w*)\b',
    'gear_shift': r'\b(?:gear|transmission|shift\s+lever|powertrain)\b.{0,160}\b(?:unintended\s+gear|wrong\s+gear|different\s+gear|loss\s+of\s+gear|shift\s+.*?neutral|gear\s+indicat\w*|loss\s+of\s+propulsion)\b',
    'door_unlatch': r'\b(?:passenger|driver|side|front|rear|vehicle)\s+doors?\b.{0,90}\b(?:open\w*|unlatch\w*|fail\w*\s+to\s+latch)\b',
    'steering_heavy': r'\b(?:steering|power\s+steering)\b.{0,110}\b(?:loss\s+of\s+assist|assist\s+loss|hard\s+to\s+steer|difficult\s+to\s+steer)\b',
    'steering_vibration': r'\b(?:steering|wheel|brake)\b.{0,80}\b(?:vibrat\w*|shak\w*|wobbl\w*|judder\w*)\b',
    'light_flicker': r'\b(?:headlamp|headlight|dashboard\s+light|exterior\s+light|lamp)\b.{0,90}\b(?:flicker\w*|flash\w*|blink\w*|intermittent\s+outage)\b',
    'ac_noise': r'\b(?:air\s+conditioning|hvac|blower\s+motor)\b.{0,80}\b(?:squeak\w*|nois\w*|rattl\w*)\b',
    'audio_disconnect': r'\b(?:bluetooth|audio|infotainment)\b.{0,90}\b(?:disconnect\w*|cut\w*\s*out|dropp\w*)\b',
}
# Weak *component-mechanism* connections. These are plausible related issues,
# not evidence that the source explicitly lists the user's symptoms.
MECHANISM_PATTERNS = {
    'engine_stall': r'\b(?:fuel\s+pump\b.{0,160}\b(?:fail\w*|defect\w*)|engine\s+harness\b.{0,100}\b(?:damag\w*|fuse))\b',
    'brake_assist': r'\b(?:brake\s+booster\s+vacuum\s+pump|vacuum/power\s+brake\s+assist|loss\s+of\s+braking\s+assist|vacuum\s+pump\s+may\s+(?:break|fail))\b',
    'airbag_warning': r'\b(?:occupant\s+classif\w*|seat\s+weight\s+sensor|passenger\s+seat\s+weight\s+sensor)\b',
    'gear_shift': r'\b(?:transmission|shift\s+lever)\b.{0,100}\b(?:unintended|different\s+gear|gear\s+position|gear\s+indication)\b',
}


def _first_match(pattern: str, source: str) -> str | None:
    m = re.search(pattern, source, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r'\s+', ' ', m.group(0)).strip()[:220] if m else None


def verify_evidence(query: str, document: dict) -> EvidenceDecision:
    """Verify a recall document against symptom evidence without any gold IDs.

    Source may be a corpus document (`text`, `component`) or NHTSA record.
    Abstain when the complaint is vague or evidence is insufficient; do not use
    this as diagnosis, VIN validation, or a guaranteed relevance classifier.
    """
    if not isinstance(query, str) or not query.strip():
        return EvidenceDecision('abstain', (), (), 'empty_query')
    text = str(document.get('text') or '').strip()
    if not text:
        return EvidenceDecision('abstain', (), (), 'missing_recall_narrative')
    signals = tuple(name for name, pattern in QUERY_SIGNALS.items()
                    if re.search(pattern, query, flags=re.IGNORECASE | re.DOTALL))
    if not signals:
        return EvidenceDecision('abstain', (), (), 'unsupported_or_vague_symptom')
    direct, related = [], []
    for name in signals:
        match = _first_match(DIRECT_PATTERNS[name], text)
        if match:
            direct.append((name, match))
        elif name in MECHANISM_PATTERNS:
            match = _first_match(MECHANISM_PATTERNS[name], text)
            if match:
                related.append((name, match))
    if direct:
        return EvidenceDecision('direct', tuple(n for n, _ in direct),
                                tuple(s for _, s in direct), 'documented_failure_mode_match')
    if related:
        return EvidenceDecision('mechanism', tuple(n for n, _ in related),
                                tuple(s for _, s in related), 'component_mechanism_only_symptom_not_explicit')
    return EvidenceDecision('abstain', signals, (), 'no_supported_failure_mode')


def filter_ranked_evidence(query: str, ranked_candidates: list[dict],
                           docs_by_id: dict[str, dict], *, top_k: int = 3,
                           allow_mechanism: bool = False) -> tuple[list[dict], list[dict]]:
    """Apply the evidence gate AFTER candidate generation/ranking.

    This never expands the candidate set. A missing document is an error, not
    permission to fabricate a source; duplicate candidate IDs are rejected.
    """
    if top_k < 1:
        raise ValueError('top_k must be positive')
    results, decisions, seen = [], [], set()
    for candidate in ranked_candidates:
        ident = str(candidate['campaign_id'])
        if ident in seen:
            raise ValueError(f'duplicate ranked campaign: {ident}')
        seen.add(ident)
        if ident not in docs_by_id:
            raise ValueError(f'campaign not in scoped corpus: {ident}')
        doc = docs_by_id[ident]
        decided = verify_evidence(query, doc)
        accepted = (decided.tier == 'direct' or
                    (decided.tier == 'mechanism' and allow_mechanism))
        decisions.append({'campaign_id': ident, 'tier': decided.tier,
                          'accepted': accepted, 'reason': decided.reason,
                          'matched_symptoms': list(decided.matched_symptoms),
                          'supported_by': list(decided.supported_by)})
        if accepted and len(results) < top_k:
            results.append({**doc, 'evidence_tier': decided.tier,
                            'evidence_snippets': list(decided.supported_by)})
    return results, decisions
