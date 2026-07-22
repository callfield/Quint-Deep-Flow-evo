"""Section-level region summary table builders."""

from __future__ import annotations

import re

import pandas as pd


BASE_COLUMNS = ["animal_id", "section_id", "region_id", "region_name", "hemisphere", "region_area_um2"]


def _natural_key(value: object) -> list[object]:
    parts = re.split(r"(\d+)", str(value))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def _count_label(value: object) -> str:
    label = str(value).strip()
    if not label:
        label = "unknown"
    return re.sub(r"\s+", "_", label)


def _channel_labels_from_section_channel_summary(section_channel_summary: pd.DataFrame | None) -> list[str]:
    if section_channel_summary is None or section_channel_summary.empty or "image_channel" not in section_channel_summary.columns:
        return []
    labels = {
        _count_label(value)
        for value in section_channel_summary["image_channel"].dropna().astype(str).tolist()
        if str(value).strip()
    }
    return sorted(labels, key=_natural_key)


def build_section_region_summary(
    region_summary: pd.DataFrame,
    section_channel_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one row per section/region with area and per-channel positive-cell counts."""

    channel_labels = _channel_labels_from_section_channel_summary(section_channel_summary)
    count_columns = [f"{label}_n_cells" for label in channel_labels]
    if region_summary is None or region_summary.empty:
        return pd.DataFrame(columns=[*BASE_COLUMNS, *count_columns])

    frame = region_summary.copy()
    for column in BASE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    if "n_cells" not in frame.columns:
        frame["n_cells"] = 0
    if "channel_or_combination" in frame.columns:
        frame["_section_count_label"] = frame["channel_or_combination"].fillna("").astype(str).map(_count_label)
    elif "image_channel" in frame.columns:
        frame["_section_count_label"] = frame["image_channel"].fillna("").astype(str).map(_count_label)
    else:
        frame["_section_count_label"] = "n_cells"

    frame["region_id"] = pd.to_numeric(frame["region_id"], errors="coerce").fillna(0).astype(int)
    frame["region_area_um2"] = pd.to_numeric(frame["region_area_um2"], errors="coerce").fillna(0.0)
    frame["n_cells"] = pd.to_numeric(frame["n_cells"], errors="coerce").fillna(0).astype(int)
    frame = frame.loc[frame["region_id"].ne(0)].copy()
    if frame.empty:
        return pd.DataFrame(columns=[*BASE_COLUMNS, *count_columns])

    id_columns = ["animal_id", "section_id", "region_id", "region_name", "hemisphere"]
    area_frame = (
        frame.groupby(id_columns, dropna=False, as_index=False)["region_area_um2"]
        .max()
        .sort_values(id_columns)
    )

    counts = (
        frame.groupby([*id_columns, "_section_count_label"], dropna=False)["n_cells"]
        .sum()
        .unstack("_section_count_label", fill_value=0)
        .reset_index()
    )
    observed_labels = [
        str(column)
        for column in counts.columns
        if column not in id_columns and str(column).strip()
    ]
    all_labels = sorted(set(channel_labels) | set(observed_labels), key=_natural_key)
    rename_map = {label: f"{label}_n_cells" for label in observed_labels}
    counts = counts.rename(columns=rename_map)
    for label in all_labels:
        column = f"{label}_n_cells"
        if column not in counts.columns:
            counts[column] = 0

    output = area_frame.merge(counts, on=id_columns, how="left")
    ordered_count_columns = [f"{label}_n_cells" for label in all_labels]
    for column in ordered_count_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0).astype(int)
    output = output[[*BASE_COLUMNS, *ordered_count_columns]]
    return output.sort_values(
        ["animal_id", "section_id", "region_id", "hemisphere"],
    ).reset_index(drop=True)
