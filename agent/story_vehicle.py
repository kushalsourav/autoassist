"""Pure, testable vehicle-context parsing for the story-first chat."""
import re

# Explicit vehicle identity, never inferred from symptoms or seeded defaults.
MAKES = {
    'honda': ('accord', 'civic', 'cr-v', 'pilot', 'fit'),
    'toyota': ('camry', 'corolla', 'rav4', 'prius', 'highlander'),
    'ford': ('f-150', 'f150', 'escape', 'focus', 'fusion', 'explorer'),
    'chevrolet': ('silverado 1500', 'silverado', 'equinox', 'malibu', 'cruze'),
    'chevy': ('silverado 1500', 'silverado', 'equinox', 'malibu', 'cruze'),
    'hyundai': ('sonata', 'elantra', 'tucson'),
    'nissan': ('altima', 'rogue', 'sentra', 'pathfinder'),
    'mercedes-benz': ('c-class', 'e-class', 'glc', 'gle'),
    'mercedes': ('c-class', 'e-class', 'glc', 'gle'),
    'bmw': ('3 series', '5 series', 'x3', 'x5'),
    'kia': ('optima', 'sorento', 'sportage'),
}

_SYMPTOM = re.compile(
    r'\b(wip\w*|camera|light|engine|brak\w*|stall\w*|power|accelera\w*|'
    r'vibrat\w*|noise|smok\w*|fail\w*|work\w*|stop\w*|leak\w*|'
    r'recall|warning|problem|squeal\w*|steer\w*|sputter\w*)\b', re.I
)


def extract_vehicle(text: str, previous: dict | None = None) -> dict:
    """Conservative deterministic extraction. Partial details persist across turns."""
    current = dict(previous or {})
    # Only explicit years. Do not assume a model year from the system clock.
    years = re.findall(r'\b(?:19[8-9]\d|20[0-2]\d)\b', text)
    if years:
        current['year'] = int(years[-1])
    lowered = text.lower()
    for make in sorted(MAKES, key=len, reverse=True):
        if re.search(r'(?<![\w-])' + re.escape(make) + r'(?![\w-])', lowered):
            canonical = {'chevy': 'Chevrolet', 'mercedes': 'Mercedes-Benz',
                         'mercedes-benz': 'Mercedes-Benz'}.get(make, make.title())
            if current.get('make') != canonical:
                current.pop('model', None)
            current['make'] = canonical
            break
    make_key = (current.get('make') or '').lower()
    if make_key in MAKES:
        models = sorted(MAKES[make_key], key=len, reverse=True)
        for model in models:
            if re.search(r'(?<![\w-])' + re.escape(model) + r'(?![\w-])', lowered):
                current['model'] = 'F-150' if model == 'f150' else model.title()
                break
    return current


def missing_vehicle_fields(vehicle: dict) -> list[str]:
    return [key for key in ('make', 'model', 'year') if not vehicle.get(key)]


def vehicle_question(vehicle: dict) -> str:
    missing = missing_vehicle_fields(vehicle)
    if len(missing) == 3:
        return ("I can help investigate that. What vehicle are you driving—"
                "the make, model and model year? You can just say something like '2018 Honda Accord'.")
    names = {'make': 'manufacturer', 'model': 'model', 'year': 'model year'}
    return ("Got it. To check recalls for the right vehicle, what's the "
            + ' and '.join(names[field] for field in missing) + '?')


