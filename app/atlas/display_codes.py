"""Stable display-code helpers for region + hemisphere labeling."""

from __future__ import annotations

import pandas as pd

from data_models.models import AtlasRegion


INT32_MAX = 2_147_483_647


def midline_offset(max_region_id: int) -> int:
    """Return the positive offset used to encode midline labels."""

    offset = 10 ** len(str(int(max_region_id)))
    if offset + int(max_region_id) <= INT32_MAX:
        return offset
    fallback = int(max_region_id) + 1
    if fallback + int(max_region_id) <= INT32_MAX:
        return fallback
    raise ValueError("Region IDs are too large to encode safely as int32 display codes")


def stable_region_display_code(region_id: int, hemisphere: str, midline_code_offset: int) -> int:
    """Encode atlas region id and hemisphere into one stable signed integer code."""

    if int(region_id) <= 0:
        return 0
    hemisphere_name = str(hemisphere).lower()
    if hemisphere_name == "left":
        return -int(region_id)
    if hemisphere_name == "right":
        return int(region_id)
    if hemisphere_name == "midline":
        return int(region_id) + int(midline_code_offset)
    return int(region_id)


def build_display_codebook(regions: dict[int, AtlasRegion]) -> pd.DataFrame:
    """Create a run-global region/hemisphere codebook."""

    max_region_id = max(regions) if regions else 0
    midline_code_offset = midline_offset(max_region_id)
    rows: list[dict[str, object]] = []
    for region_id, region in sorted(regions.items()):
        for hemisphere in ("midline", "left", "right"):
            rows.append(
                {
                    "display_code": stable_region_display_code(region_id, hemisphere, midline_code_offset),
                    "region_id": region_id,
                    "region_name": region.name,
                    "hemisphere": hemisphere,
                    "parent_region_id": region.parent_id,
                    "hierarchy": " > ".join(region.hierarchy_names),
                    "midline_code_offset": midline_code_offset,
                    "encoding_rule": "right=+region_id,left=-region_id,midline=region_id+midline_code_offset",
                }
            )
    return pd.DataFrame(rows)
