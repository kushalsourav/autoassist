"""NHTSA US-market vehicle recall lookup. Recall hits are not verified diagnoses."""
import httpx

API_URL = 'https://api.nhtsa.gov/recalls/recallsByVehicle'


async def fetch_recalls(make: str, model: str, year: int) -> tuple[list[dict], str]:
    params = {'make': make, 'model': model, 'modelYear': year}
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            response = await client.get(API_URL, params=params)
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get('results'), list):
            return [], 'unavailable'
        source_url = str(response.url)
        output = []
        for item in payload['results'][:25]:
            if not isinstance(item, dict):
                continue
            number = str(item.get('NHTSACampaignNumber') or 'Unknown recall')
            component = str(item.get('Component') or 'Vehicle component')
            summary = str(item.get('Summary') or '')
            remedy = str(item.get('Remedy') or '')
            output.append({
                'id': number,
                'component': component[:300],
                'text': f'{component}. {summary} Remedy: {remedy}'[:4000],
                'url': source_url,
            })
        return output, 'ok'
    except (httpx.HTTPError, ValueError, TypeError):
        return [], 'unavailable'
