"""Parsers for QuickNII / VisuAlign JSON registration files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_models.models import RegistrationData, RegistrationSlice


def _normalize_registration_filename(name: str) -> str:
    return Path(name).stem.strip().lower().replace(" ", "_")


def _parse_anchoring(raw_slice: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchoring = raw_slice.get("anchoring")
    if isinstance(anchoring, dict):
        origin = np.asarray(anchoring.get("origin", (0, 0, 0)), dtype=np.float64)
        u = np.asarray(anchoring.get("u", (0, 0, 0)), dtype=np.float64)
        v = np.asarray(anchoring.get("v", (0, 0, 0)), dtype=np.float64)
        return origin, u, v
    if isinstance(anchoring, list) and len(anchoring) >= 9:
        values = np.asarray(anchoring[:9], dtype=np.float64)
        return values[0:3], values[3:6], values[6:9]
    raise ValueError(f"Unsupported anchoring payload in slice {raw_slice.get('filename')!r}")


def parse_registration_file(path: Path) -> RegistrationData:
    """Parse a QuickNII / VisuAlign registration JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if isinstance(raw, list):
        slices_raw = raw
        target = ""
        target_resolution = None
    else:
        slices_raw = raw.get("slices", [])
        target = raw.get("target", "")
        target_resolution = tuple(raw.get("target-resolution", ())) or None

    slices: list[RegistrationSlice] = []
    for item in slices_raw:
        if not isinstance(item, dict) or "filename" not in item:
            continue
        origin, u, v = _parse_anchoring(item)
        slices.append(
            RegistrationSlice(
                filename=item["filename"],
                nr=item.get("nr"),
                width=int(item.get("width", 0)),
                height=int(item.get("height", 0)),
                origin=origin,
                u=u,
                v=v,
                target_resolution=target_resolution,
                markers=item.get("markers", []),
                raw=item,
            )
        )

    return RegistrationData(
        source_path=path,
        target=target,
        target_resolution=target_resolution,
        slices=slices,
        raw=raw if isinstance(raw, dict) else {"slices": raw},
    )


def match_registration_slice(registration: RegistrationData, image_name: str) -> RegistrationSlice:
    """Return the registration entry that matches an image filename."""

    lookup = _normalize_registration_filename(image_name)
    for item in registration.slices:
        if _normalize_registration_filename(item.filename) == lookup:
            return item
    raise KeyError(f"No registration slice matched image {image_name!r} in {registration.source_path}")
