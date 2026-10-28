"""Vehicle-aware, *permissive* intent routing for AutoAssist.

This router is a lightweight UX gate, NOT a safety classifier or a mechanical
classifier. When a vehicle is selected, do not discard detailed complaints
simply because their component names are missing from a keyword list.

Mechanical component prediction is the job of the trained ML model. Out-of-
scope handling protects resources but should not suppress valid complaints.
"""
import re

# High-confidence signals. This is deliberately not an exhaustive whitelist.
_AUTOMOTIVE = re.compile(
    r'\b(car|cars|vehicle|vehicles|automobile|sedan|truck|suv|engine|transmission|'
    r'brake|brakes|airbag|dashboard|headlights?|battery|alternator|stalls?|'
    r'accelerat\w*|ignition|powertrain|recall|nhtsa|vin|mileage|odometer|'
    r'check.engine|warning light|steering|tires?|tyres?|clutch|fuel|exhaust|'
    r'diagnostic|dtc|obd|repair|mechanic|honda|toyota|mercedes|bmw|ford|'
    r'volkswagen|hyundai|kia|audi|nissan|windshield|windscreen|wipers?|'
    r'rearview|rear.view|backup camera|reverse camera|rear camera|'
    r'defrost\w*|side.mirror|door lock|key fob|infotainment|'
    r'radiator|coolant|suspension|abs|bcm|ecu|tail.?lights?|'
    r'parking sensors?)\b', re.I)

# Strong non-auto examples. Mixed topical signals still require clarification.
_NON_AUTO = re.compile(
    r'\b(tigers?|lions?|elephants?|football|cricket|movie|movies|'
    r'recipes?|cooking|weather|stock market|politics|election|'
    r'jokes?|poems?|haiku|javascript|programming|calculus)\b', re.I)

_VAGUE_CARS = re.compile(
    r'\b(how many (cars|vehicles)|tell (me )?about (the )?cars|'
    r'what about (the )?cars)\b', re.I)
_FOLLOWUP = re.compile(
    r'\b(it|that|those|this|same|earlier|previous|safe to drive|'
    r'what should i do|any more|what about|why|how)\b', re.I)

# Short messages without evidence of vehicle intent should still be clarified.
# For detailed input with selected make/model/year, prefer the automotive
# pipeline over an incomplete vocabulary/keyword gate.
_MIN_CONTEXT_WORDS = 5


def _has_vehicle_context(vehicle: dict | None) -> bool:
    """Require filled structured vehicle details (not an arbitrary dictionary)."""
    if not isinstance(vehicle, dict):
        return False
    make = vehicle.get('make')
    model = vehicle.get('model')
    year = vehicle.get('year')
    return (isinstance(make, str) and bool(make.strip())
            and isinstance(model, str) and bool(model.strip())
            and isinstance(year, int) and not isinstance(year, bool)
            and 1980 <= year <= 2030)


def route_intent(message: str, history: list[dict] | None = None,
                 vehicle: dict | None = None) -> str:
    """Return automotive, out_of_scope or clarification.

    Precedence: explicit out-of-scope > vague counts > automotive cues >
    relevant conversation follow-up > sufficiently detailed vehicle-context
    request > clarification. No ML classification happens in this function.

    Since the frontend may prefill vehicle information, the contextual fallback
    is deliberately restricted to substantive messages. This is a permissive
    recall/complaint UX heuristic, not a certified semantic intent classifier.
    """
    text = (message or '').strip()
    if not text:
        return 'clarification'
    automotive = bool(_AUTOMOTIVE.search(text))
    unrelated = bool(_NON_AUTO.search(text))
    if unrelated:
        return 'clarification' if automotive else 'out_of_scope'
    if _VAGUE_CARS.search(text) and not re.search(
            r'\b(recall|dataset|nhtsa|production|registered)\b', text, re.I):
        return 'clarification'
    if automotive:
        return 'automotive'

    prior_user = [entry.get('content', '') for entry in (history or [])
                  if entry.get('role') == 'user']
    if (prior_user and any(_AUTOMOTIVE.search(p) for p in prior_user)
            and _FOLLOWUP.search(text)):
        return 'automotive'

    if _has_vehicle_context(vehicle) and len(re.findall(r"\b[\w']+\b", text)) >= _MIN_CONTEXT_WORDS:
        return 'automotive'
    return 'clarification'


def intent_reply(intent: str) -> str:
    if intent == 'out_of_scope':
        return ('AutoAssist focuses on vehicle complaints, recalls, and automotive safety. '
                'I can help investigate an issue with your car, but I am not a general-purpose assistant.')
    return ('Could you clarify what you want to know about the vehicle? '
            'For example, share the symptoms, a recall question, or the dataset and metric '
            'you mean by a vehicle count.')
