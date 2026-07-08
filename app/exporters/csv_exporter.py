"""DataFrame conversion and CSV export helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_models.models import DetectedObject


def cell_rows_from_objects(
    objects: list[DetectedObject],
    coordinate_unit: str,
    analysis_channels: list[str],
    include_patch_ids: bool = False,
) -> list[dict[str, object]]:
    """Flatten detected objects into export rows."""

    rows: list[dict[str, object]] = []
    channel_columns = [channel.upper() for channel in analysis_channels if channel and channel.strip()]
    for obj in objects:
        allen = obj.allen_pir_um or (None, None, None)
        row = {
            "animal_id": obj.animal_id,
            "section_id": obj.section_id,
            "image_channel": obj.image_channel,
            "cell_id": obj.cell_id,
            "centroid_x_px": obj.centroid_x_px,
            "centroid_y_px": obj.centroid_y_px,
            "allen_posterior_um": allen[0],
            "allen_inferior_um": allen[1],
            "allen_right_um": allen[2],
            "AP": obj.ap,
            "ML": obj.ml,
            "DV": obj.dv,
            "hemisphere": obj.hemisphere,
            "region_id": obj.region_id,
            "region_name": obj.region_name,
            "hierarchy": " > ".join(obj.hierarchy_names),
            "area_px": obj.area_px,
            "mean_intensity": obj.mean_intensity,
            "integrated_intensity": obj.integrated_intensity,
            "assignment_fraction": obj.assignment_fraction,
            "overlap_group": obj.overlap_group,
        }
        if include_patch_ids:
            row["overlap_group_id"] = obj.overlap_group_id
            row["atlas_display_code"] = obj.atlas_display_code
            row["atlas_patch_component"] = obj.atlas_patch_component
            row["atlas_patch_id"] = obj.atlas_patch_id
        matched = {channel.upper() for channel in obj.matched_channels}
        for channel in channel_columns:
            row[channel] = 1 if channel in matched else 0
        rows.append(row)
    return rows


def write_tables(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    """Write all dataframes as CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in tables.items():
        frame.to_csv(output_dir / filename, index=False)


def metrics_json(metrics: dict[str, object]) -> str:
    """Compact JSON serializer used in section summary tables."""

    return json.dumps(metrics, sort_keys=True, ensure_ascii=False)
