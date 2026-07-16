"""Persistent cache for section-level registered atlas maps."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from atlas.repository import AtlasRepository
from data_models.models import RegistrationSlice

CACHE_VERSION = 2


@dataclass(slots=True)
class RegisteredSectionCacheEntry:
    """Atlas maps that can be reused across channels and repeated runs."""

    region_map: np.ndarray
    hemisphere_map: np.ndarray
    qc_metrics: dict[str, Any]
    cache_path: Path | None = None


def build_registered_section_cache_key(
    *,
    registration_slice: RegistrationSlice,
    output_shape: tuple[int, int],
    atlas: AtlasRepository,
    processing_fingerprint: dict[str, Any],
) -> str:
    """Create a stable cache key for one registered atlas map."""

    payload = {
        "cache_version": CACHE_VERSION,
        "registration_slice": _registration_slice_fingerprint(registration_slice),
        "output_shape": [int(output_shape[0]), int(output_shape[1])],
        "atlas_labels": _file_fingerprint(atlas.config.labels_path),
        "atlas_name": atlas.config.name,
        "atlas_voxel_size_um": float(atlas.voxel_size_um),
        "atlas_resolution_vox": [int(value) for value in atlas.require_labels().shape],
        "processing": processing_fingerprint,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return digest


def build_registered_section_cache_path(
    cache_dir: Path,
    *,
    animal_id: str,
    section_id: str,
    cache_key: str,
) -> Path:
    """Return the concrete cache file path for one section map."""

    safe_animal = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in animal_id)
    safe_section = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in section_id)
    return cache_dir / safe_animal / f"{safe_animal}_{safe_section}_{cache_key}.npz"


def load_registered_section_cache(cache_path: Path) -> RegisteredSectionCacheEntry:
    """Load a cached section atlas map from disk."""

    with np.load(cache_path, allow_pickle=False) as payload:
        region_map = np.asarray(payload["region_map"], dtype=np.uint32)
        hemisphere_map = np.asarray(payload["hemisphere_map"], dtype=np.int8)
        qc_metrics = json.loads(str(payload["qc_metrics_json"].item()))
    return RegisteredSectionCacheEntry(
        region_map=region_map,
        hemisphere_map=hemisphere_map,
        qc_metrics=qc_metrics,
        cache_path=cache_path,
    )


def save_registered_section_cache(
    cache_path: Path,
    *,
    region_map: np.ndarray,
    hemisphere_map: np.ndarray,
    qc_metrics: dict[str, Any],
) -> None:
    """Persist one section atlas map to disk for later reuse."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp-{os.getpid()}")
    np.savez(
        temp_path,
        region_map=np.asarray(region_map, dtype=np.uint32),
        hemisphere_map=np.asarray(hemisphere_map, dtype=np.int8),
        qc_metrics_json=np.asarray(json.dumps(qc_metrics, sort_keys=True), dtype=np.str_),
    )
    Path(f"{temp_path}.npz").replace(cache_path)


def _file_fingerprint(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    hash_size = 1_048_576
    digest = hashlib.sha1()
    with resolved.open("rb") as handle:
        if stat.st_size <= hash_size * 2:
            digest.update(handle.read())
        else:
            digest.update(handle.read(hash_size))
            handle.seek(max(0, stat.st_size - hash_size))
            digest.update(handle.read(hash_size))
    return {
        "name": resolved.name,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha1_head_tail": digest.hexdigest(),
    }


def _registration_slice_fingerprint(registration_slice: RegistrationSlice) -> dict[str, Any]:
    """Build a stable fingerprint for a single registration entry."""

    raw_payload = registration_slice.raw if isinstance(registration_slice.raw, dict) else {}
    target_resolution = registration_slice.target_resolution
    return {
        "filename": str(registration_slice.filename),
        "nr": int(registration_slice.nr) if registration_slice.nr is not None else None,
        "width": int(registration_slice.width),
        "height": int(registration_slice.height),
        "target_resolution": [int(value) for value in target_resolution] if target_resolution else None,
        "raw": raw_payload,
    }
