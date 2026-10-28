"""Diagnose AG-UI OpenRouter settings; never print API keys or raw credentials.

Usage: python -m scripts.diagnose_story_llm [--probe]
--probe makes one short OpenRouter test call; omitting it makes no network calls.
"""
from __future__ import annotations

import argparse
import asyncio

def explain_status(code: int) -> str:
    if code in (401, 403):
        return "Authentication/permission failure: check OPENROUTER_API_KEY."
    if code == 404:
        return "Model not available: check OPENROUTER_MODEL name and provider access."
    if code == 429:
        return "Rate limit or quota reached. Try again later or another permitted model."
    if code >= 500:
        return "OpenRouter/provider temporary server failure."
    if code >= 400:
        return "Request rejected; inspect OpenRouter model availability and request settings."
    return "HTTP request accepted; the actual model may still return empty content."


async def probe(settings) -> None:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {settings.openrouter_api_key}",
                         "Content-Type": "application/json"},
                json={"model": settings.openrouter_model, "temperature": 0,
                      "max_tokens": 64,
                      "messages": [{"role": "user", "content": "Reply with the word ready."}]},
            )
        print(f"OpenRouter probe status: HTTP {response.status_code}")
        print(explain_status(response.status_code))
        if response.status_code == 200:
            try:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content")
                print("Model returned nonempty text:", bool(content and str(content).strip()))
            except (ValueError, KeyError, TypeError, IndexError):
                print("The provider response was not in the expected chat-completions format.")
    except httpx.TimeoutException:
        print("The OpenRouter probe timed out after 25 seconds.")
    except httpx.RequestError as exc:
        print(f"OpenRouter connection failed: {type(exc).__name__}")


def main() -> None:
    from backend.app.config import settings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true", help="Send one live OpenRouter request (may incur API usage)")
    args = parser.parse_args()
    print("OpenRouter configured:", bool(settings.openrouter_api_key))
    print("Model:", settings.openrouter_model or "(missing)")
    print("Timeout seconds:", settings.openrouter_timeout_seconds)
    print("Security: API key is intentionally never printed")
    if not settings.openrouter_api_key or not settings.openrouter_model:
        print("Set OPENROUTER_API_KEY and OPENROUTER_MODEL in the PROJECT ROOT .env; restart port 8123.")
        return
    if args.probe:
        asyncio.run(probe(settings))
    else:
        print("No network request made. Add --probe to check provider status.")


if __name__ == "__main__":
    main()
