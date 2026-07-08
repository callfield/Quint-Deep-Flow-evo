"""Lightweight image loading helpers built on Pillow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def load_image_array(path: Path, grayscale: bool = False) -> np.ndarray:
    """Load an image as a numpy array."""

    with Image.open(path) as image:
        if grayscale:
            image = image.convert("L")
        elif image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")
        return np.asarray(image)


def load_mask_array(path: Path, threshold: int = 1) -> np.ndarray:
    """Load a mask image as a boolean array."""

    mask = load_image_array(path, grayscale=True)
    unique_values = np.unique(mask)
    if unique_values.size == 2 and int(unique_values[0]) == 0 and int(unique_values[1]) == 255:
        return np.asarray(mask == 255, dtype=bool)
    if threshold <= 1 and unique_values.size <= 8:
        foreground_value, _ = infer_mask_foreground_value(mask)
        return np.asarray(mask == foreground_value, dtype=bool)
    return mask >= threshold


def infer_mask_foreground_value(mask: np.ndarray, min_area_px: int = 1, max_area_px: int = 0) -> tuple[int, list[dict[str, float | int]]]:
    """Infer which discrete label in an ilastik-style mask corresponds to cells."""

    values = [int(value) for value in np.unique(mask)]
    if not values:
        return 0, []

    candidate_stats: list[dict[str, float | int]] = []
    structure = np.ones((3, 3), dtype=np.uint8)
    min_area = max(1, int(min_area_px))
    max_area = int(max_area_px)

    for value in values:
        binary = np.asarray(mask == value, dtype=bool)
        labeled, _ = ndimage.label(binary, structure=structure)
        areas = np.bincount(labeled.ravel())[1:]
        if areas.size:
            plausible = areas >= min_area
            if max_area > 0:
                plausible &= areas <= max_area
            plausible_areas = areas[plausible]
            stats = {
                "value": int(value),
                "foreground_fraction": float(binary.mean()),
                "component_count": int(areas.size),
                "plausible_component_count": int(plausible_areas.size),
                "plausible_area_sum": int(plausible_areas.sum()) if plausible_areas.size else 0,
                "max_component_area": int(areas.max()),
            }
        else:
            stats = {
                "value": int(value),
                "foreground_fraction": float(binary.mean()),
                "component_count": 0,
                "plausible_component_count": 0,
                "plausible_area_sum": 0,
                "max_component_area": 0,
            }
        candidate_stats.append(stats)

    if any(int(stats["plausible_component_count"]) > 0 for stats in candidate_stats):
        best = max(
            candidate_stats,
            key=lambda stats: (
                int(stats["plausible_component_count"]),
                int(stats["plausible_area_sum"]),
                -float(stats["foreground_fraction"]),
                int(stats["component_count"]),
            ),
        )
    else:
        best = min(
            candidate_stats,
            key=lambda stats: (
                float(stats["foreground_fraction"]),
                -int(stats["component_count"]),
                -int(stats["plausible_area_sum"]),
            ),
        )
    return int(best["value"]), candidate_stats


def normalize_mask_to_binary(mask: np.ndarray, foreground_value: int) -> np.ndarray:
    """Convert a discrete-label mask into 0/255 binary form with cells white and background black."""

    binary = np.asarray(mask == int(foreground_value), dtype=np.uint8)
    return np.where(binary > 0, 255, 0).astype(np.uint8, copy=False)


def normalize_ilastik_mask_file_inplace(path: Path, min_area_px: int = 1, max_area_px: int = 0) -> dict[str, object]:
    """Infer the ilastik foreground class, rewrite the file as 0/255, and return a report row."""

    mask = load_image_array(path, grayscale=True)
    foreground_value, candidate_stats = infer_mask_foreground_value(mask, min_area_px=min_area_px, max_area_px=max_area_px)
    normalized = normalize_mask_to_binary(mask, foreground_value)
    Image.fromarray(normalized, mode="L").save(path)

    chosen_stats = next((stats for stats in candidate_stats if int(stats["value"]) == int(foreground_value)), {})
    return {
        "mask_file": str(path),
        "original_unique_values": ";".join(str(int(value)) for value in np.unique(mask)),
        "chosen_foreground_value": int(foreground_value),
        "chosen_foreground_fraction": float(chosen_stats.get("foreground_fraction", 0.0) or 0.0),
        "chosen_component_count": int(chosen_stats.get("component_count", 0) or 0),
        "chosen_plausible_component_count": int(chosen_stats.get("plausible_component_count", 0) or 0),
        "min_area_px": int(min_area_px),
        "max_area_px": int(max_area_px),
        "normalized_unique_values": "0;255",
    }


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a grayscale or RGBA image array into RGB."""

    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def grayscale_intensity(image: np.ndarray) -> np.ndarray:
    """Create a grayscale intensity image from RGB or grayscale input."""

    if image.ndim == 2:
        return image.astype(np.float32, copy=False)
    rgb = ensure_rgb(image).astype(np.float32, copy=False)
    return np.dot(rgb[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
