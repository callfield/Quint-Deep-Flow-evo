"""Comparison against existing Nutil coordinate outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from atlas.repository import AtlasRepository
from config.settings import ComparisonConfig
from data_models.models import SectionChannelResult
from quantification.hemisphere import hemisphere_from_ml_um


def _load_nutil_coordinate_json(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    rows: list[dict[str, object]] = []
    points: list[np.ndarray] = []
    for item in raw:
        triplets = np.asarray(item.get("triplets", []), dtype=np.float32)
        reshaped = triplets.reshape((-1, 3)) if triplets.size else np.empty((0, 3), dtype=np.float32)
        rows.append({"region_name": item.get("name", "Unknown"), "count": int(reshaped.shape[0])})
        if reshaped.size:
            points.append(reshaped)
    frame = pd.DataFrame(rows)
    point_cloud = np.concatenate(points, axis=0) if points else np.empty((0, 3), dtype=np.float32)
    return frame, point_cloud


def _find_coordinate_json(result_dir: Path | None, result: SectionChannelResult) -> Path | None:
    if result_dir is None:
        return None
    coordinates_dir = result_dir / "Coordinates"
    if not coordinates_dir.exists():
        return None
    pattern = f"*{result.bundle.section_id}*.json"
    matches = sorted(coordinates_dir.glob(pattern))
    return matches[0] if matches else None


def generate_comparison_report(
    section_results: list[SectionChannelResult],
    atlas: AtlasRepository,
    config: ComparisonConfig,
    output_dir: Path,
) -> pd.DataFrame:
    """Compare QUINTnext output with existing Nutil slice coordinate JSON files."""

    rows: list[dict[str, object]] = []
    for result in section_results:
        coord_json = _find_coordinate_json(result.bundle.existing_result_dir, result)
        if coord_json is None:
            continue
        nutil_counts, nutil_points = _load_nutil_coordinate_json(coord_json)
        our_points = np.array([obj.allen_pir_um for obj in result.detected_objects if obj.allen_pir_um is not None], dtype=np.float32)
        our_region_counts = (
            pd.DataFrame({"region_name": [obj.region_name for obj in result.detected_objects]})
            .value_counts()
            .rename("count")
            .reset_index()
        )
        merged = (
            our_region_counts.merge(nutil_counts, on="region_name", how="outer", suffixes=("_quintnext", "_nutil"))
            .fillna(0)
        )
        merged["abs_delta"] = (merged["count_quintnext"] - merged["count_nutil"]).abs()
        mean_nn_distance_um = None
        if len(our_points) and len(nutil_points):
            tree = cKDTree(nutil_points)
            distances, _ = tree.query(our_points, k=1)
            mean_nn_distance_um = float(np.mean(distances))

        nutil_hemi = {"left": 0, "right": 0, "midline": 0}
        for point in nutil_points:
            ml_um = float(point[2]) - atlas.config.allen_bregma_um[2]
            nutil_hemi[hemisphere_from_ml_um(ml_um, 75.0)] += 1
        our_hemi = pd.Series([obj.hemisphere for obj in result.detected_objects]).value_counts().to_dict()
        left_diff = int(our_hemi.get("left", 0) - nutil_hemi["left"])
        right_diff = int(our_hemi.get("right", 0) - nutil_hemi["right"])

        total_diff = int(len(result.detected_objects) - int(nutil_counts["count"].sum()))
        warning_parts: list[str] = []
        if abs(total_diff) > config.max_abs_count_delta_warning:
            warning_parts.append("large_total_count_delta")
        if mean_nn_distance_um is not None and mean_nn_distance_um > config.max_mean_nn_distance_um_warning:
            warning_parts.append("large_position_delta")
        if (merged["abs_delta"].max() if not merged.empty else 0) > config.max_abs_count_delta_warning:
            warning_parts.append("large_region_delta")

        rows.append(
            {
                "animal_id": result.bundle.animal_id,
                "section_id": result.bundle.section_id,
                "channel": result.bundle.channel,
                "coordinate_json": str(coord_json),
                "quintnext_total_cells": len(result.detected_objects),
                "nutil_total_cells": int(nutil_counts["count"].sum()),
                "total_diff": total_diff,
                "max_region_abs_diff": int(merged["abs_delta"].max()) if not merged.empty else 0,
                "mean_nn_distance_um": mean_nn_distance_um,
                "left_diff": left_diff,
                "right_diff": right_diff,
                "warning": ";".join(warning_parts),
            }
        )

    frame = pd.DataFrame(rows)
    markdown_lines = [
        "# Comparison Report",
        "",
        "This report compares QUINTnext output against existing Nutil coordinate JSON files when they are present.",
        "",
    ]
    if frame.empty:
        markdown_lines.append("No comparable Nutil coordinate JSON files were found.")
    else:
        columns = list(frame.columns)
        markdown_lines.append("| " + " | ".join(columns) + " |")
        markdown_lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in frame.fillna("").astype(str).to_dict("records"):
            markdown_lines.append("| " + " | ".join(row[column] for column in columns) + " |")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_report.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    return frame
