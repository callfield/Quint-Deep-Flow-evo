"""Shared version metadata for QUINTdeepflow desktop apps."""

from __future__ import annotations

APP_VERSION = "2026.07.16.3"


def version_label(app_name: str) -> str:
    """Return a compact user-facing version label for one desktop app."""

    return f"{app_name} v{APP_VERSION}"
