"""Hemisphere helpers."""

from __future__ import annotations


def hemisphere_from_ml_um(ml_um: float, threshold_um: float) -> str:
    """Map Bregma-centered ML into left / right labels only."""

    del threshold_um
    return "right" if ml_um >= 0.0 else "left"
