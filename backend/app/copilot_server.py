"""Compatibility import: /api and /agent are one backend app."""
from backend.app.main import app
__all__ = ["app"]
