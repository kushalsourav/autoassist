"""Download REAL US-market NHTSA owner complaints and produce weak component labels.

These labels represent reported complaint components, NOT verified mechanical faults.
No training corpus is bundled or fabricated. Internet connection required.
"""
import argparse
import csv
from pathlib import Path
import time
import httpx

ENDPOINT = 'https://api.nhtsa.gov/complaints/complaintsByVehicle'
# Starting sampling frame; expand to more years/models for credible evaluation.
VEHICLES = [
    ('Honda', 'Accord', 2018), ('Honda', 'Civic', 2018),
    ('Toyota', 'Camry', 2018), ('Toyota', 'Corolla', 2018),
    ('Ford', 'Escape', 2018), ('Ford', 'F-150', 2018),
    ('Chevrolet', 'Silverado 1500', 2018), ('Chevrolet', 'Equinox', 2018),
    ('Hyundai', 'Sonata', 2018), ('Hyundai', 'Elantra', 2018),
    ('Nissan', 'Altima', 2018), ('Nissan', 'Rogue', 2018),
]
COMPONENT_MAP = [
    (('AIR BAGS', 'SEAT BELTS'), 'Airbags / Safety'),
    (('SERVICE BRAKES', 'PARKING BRAKE'), 'Brakes'),
    (('STEERING', 'SUSPENSION'), 'Steering / Suspension'),
    (('POWER TRAIN',), 'Transmission'),
    (('ENGINE', 'FUEL/PROPULSION SYSTEM', 'VEHICLE SPEED CONTROL'), 'Engine / Powertrain'),
    (('ELECTRICAL SYSTEM', 'EXTERIOR LIGHTING'), 'Electrical'),
]


def categorize(components: str) -> str | None:
    value = components.upper()
    for tokens, category in COMPONENT_MAP:
        if any(t in value for t in tokens):
            return category
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('ml/data/nhtsa_complaints.csv'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    with httpx.Client(timeout=30) as client:
        for make, model, year in VEHICLES:
            try:
                response = client.get(ENDPOINT, params={'make': make, 'model': model, 'modelYear': year})
                response.raise_for_status()
                records = response.json().get('results', [])
            except (httpx.HTTPError, ValueError, AttributeError) as error:
                print(f'WARNING: {make} {model} {year}: {error}')
                continue
            for item in records:
                if not isinstance(item, dict):
                    continue
                summary = str(item.get('summary') or item.get('Summary') or '').strip()
                components = str(item.get('components') or item.get('Components') or '')
                category = categorize(components)
                if len(summary) < 30 or not category:
                    continue
                key = summary.lower().strip()
                if key in seen:
                    continue
                seen.add(key)
                rows.append({'summary': summary, 'category': category,
                             'reported_components': components,
                             'make': make, 'model': model, 'year': year})
            print(f'{make} {model}: {len(records)} records; {len(rows)} usable total')
            time.sleep(0.25)
    if not rows:
        raise SystemExit('No data retrieved; check NHTSA API connectivity/response keys.')
    with args.output.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f'Saved {len(rows)} labeled complaints to {args.output}')
    print('Labels are WEAK SUPERVISION from owner-reported components, not confirmed diagnosis.')


if __name__ == '__main__':
    main()
