"""Dataset loading, session persistence, and export helpers for QUINTdeepflow3."""

from __future__ import annotations

from dataclasses import asdict
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import yaml

from multichannel.matcher import parse_overlap_spec
from quantification.section_summary import build_section_region_summary
from quality_check.models import (
    OmitRegionSelection,
    OmitSessionState,
    OverlayChannelInfo,
    OverlayDataset,
    OverlaySliceInfo,
    RegionInfo,
)
from quality_check.render import component_label_map_for_code, render_qc_image


OVERLAY_PATTERN = re.compile(r"^(?P<animal>.+)_(?P<section>[^_]+)_overlay_full\.tif{1,2}$", re.IGNORECASE)
CHANNEL_COLUMN_PATTERN = re.compile(r"^CH\d+$", re.IGNORECASE)


def load_overlay_dataset(overlay_dir: Path) -> OverlayDataset:
    """Load stack metadata, channel map, and atlas-code lookup for one overlay directory."""

    overlay_dir = Path(overlay_dir).expanduser().resolve()
    if not overlay_dir.exists():
        raise FileNotFoundError(f"Overlay directory does not exist: {overlay_dir}")

    overlay_paths = sorted(
        overlay_dir.glob("*_overlay_full.tif*"),
        key=lambda path: _overlay_sort_key(path.name),
    )
    if not overlay_paths:
        raise FileNotFoundError(f"No *_overlay_full.tif* files found in: {overlay_dir}")

    channel_map_path = _single_path_or_raise(overlay_dir, "*multichannel_channel_maps.xlsx", "channel map workbook")
    channels = _load_channel_map(channel_map_path)
    codebook_path = _find_codebook(overlay_dir)
    region_lookup = _load_region_lookup(codebook_path)

    slices = [_parse_overlay_slice(path) for path in overlay_paths]
    return OverlayDataset(
        overlay_dir=overlay_dir,
        channel_map_path=channel_map_path,
        codebook_path=codebook_path,
        slices=slices,
        channels=channels,
        region_lookup=region_lookup,
    )


def load_overlay_stack(slice_info: OverlaySliceInfo):
    """Load one overlay TIFF stack into memory."""

    stack = tifffile.imread(slice_info.overlay_path)
    if stack.ndim != 3:
        raise ValueError(f"Expected 3D stack (C,H,W) in {slice_info.overlay_path}, got shape {stack.shape}")
    return stack


def create_default_session(dataset: OverlayDataset) -> OmitSessionState:
    """Create a fresh interactive session from the dataset defaults."""

    default_visible = [
        info.content
        for info in dataset.visible_channel_infos
        if (info.is_raw_image or info.is_cell_roi or info.is_outline or info.is_omit_mask)
    ]
    if not default_visible:
        default_visible = [info.content for info in dataset.visible_channel_infos]
    return OmitSessionState(
        overlay_dir=dataset.overlay_dir,
        channel_map_path=dataset.channel_map_path,
        selected_slice_key=dataset.slices[0].key if dataset.slices else "",
        visible_contents=default_visible,
        omitted_regions_by_slice={slice_info.key: [] for slice_info in dataset.slices},
    )


def session_path_for_overlay_dir(overlay_dir: Path) -> Path:
    """Return the default auto-save path for one overlay directory."""

    overlay_dir = Path(overlay_dir).expanduser().resolve()
    return overlay_dir.parent / "omitByQDF3" / "omit_session.yaml"


def save_session(session: OmitSessionState, path: Path) -> None:
    """Persist session state to YAML."""

    payload = asdict(session)
    payload["overlay_dir"] = str(session.overlay_dir)
    payload["channel_map_path"] = str(session.channel_map_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def load_session(path: Path) -> OmitSessionState:
    """Load session state from YAML."""

    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    regions = dict(raw.get("omitted_regions_by_slice", {}))
    legacy_codes = dict(raw.get("omitted_display_codes_by_slice", {}))
    normalized_regions: dict[str, list[OmitRegionSelection]] = {}
    for key, values in regions.items():
        selections: list[OmitRegionSelection] = []
        for value in values or []:
            if isinstance(value, dict):
                selections.append(
                    OmitRegionSelection(
                        display_code=int(value.get("display_code", 0)),
                        component_label=int(value.get("component_label", 0)),
                    )
                )
            else:
                selections.append(OmitRegionSelection(display_code=int(value), component_label=1))
        normalized_regions[str(key)] = selections
    for key, values in legacy_codes.items():
        if str(key) in normalized_regions:
            continue
        normalized_regions[str(key)] = [
            OmitRegionSelection(display_code=int(code), component_label=1)
            for code in (values or [])
        ]
    return OmitSessionState(
        overlay_dir=Path(raw.get("overlay_dir", "")),
        channel_map_path=Path(raw.get("channel_map_path", "")),
        selected_slice_key=str(raw.get("selected_slice_key", "")),
        visible_contents=[str(value) for value in raw.get("visible_contents", [])],
        omitted_regions_by_slice=normalized_regions,
    )


def normalize_session(dataset: OverlayDataset, session: OmitSessionState) -> OmitSessionState:
    """Align a loaded session to the current dataset contents."""

    available_contents = {info.content for info in dataset.visible_channel_infos}
    visible_contents = [content for content in session.visible_contents if content in available_contents]
    if not visible_contents:
        visible_contents = create_default_session(dataset).visible_contents

    valid_slice_keys = {slice_info.key for slice_info in dataset.slices}
    omitted: dict[str, list[OmitRegionSelection]] = {}
    for slice_info in dataset.slices:
        existing = session.omitted_regions_by_slice.get(slice_info.key, [])
        normalized_existing: list[OmitRegionSelection] = []
        for selection in existing:
            normalized_existing.append(
                OmitRegionSelection(
                    display_code=int(selection.display_code),
                    component_label=max(1, int(selection.component_label)),
                )
            )
        omitted[slice_info.key] = normalized_existing

    selected_slice_key = session.selected_slice_key if session.selected_slice_key in valid_slice_keys else (dataset.slices[0].key if dataset.slices else "")
    return OmitSessionState(
        overlay_dir=dataset.overlay_dir,
        channel_map_path=dataset.channel_map_path,
        selected_slice_key=selected_slice_key,
        visible_contents=visible_contents,
        omitted_regions_by_slice=omitted,
    )


def build_omit_rows(dataset: OverlayDataset, session: OmitSessionState) -> list[dict[str, Any]]:
    """Build tabular omit rows for CSV export."""

    raw_contents = [content for content in session.visible_contents if content.endswith("_raw_image")]
    raw_summary = ";".join(raw_contents)
    rows: list[dict[str, Any]] = []
    for slice_info in dataset.slices:
        selections = sorted(
            session.omitted_regions_by_slice.get(slice_info.key, []),
            key=lambda item: (int(item.display_code), int(item.component_label)),
        )
        for selection in selections:
            region = region_info_for_code(dataset.region_lookup, int(selection.display_code))
            rows.append(
                {
                    "animal_id": slice_info.animal_id,
                    "section_id": slice_info.section_id,
                    "overlay_file": str(slice_info.overlay_path),
                    "display_code": int(selection.display_code),
                    "component_label": int(selection.component_label),
                    "atlas_patch_id": _format_patch_id(
                        slice_info.animal_id,
                        slice_info.section_id,
                        int(selection.display_code),
                        int(selection.component_label),
                    ),
                    "region_id": int(region.region_id),
                    "region_name": region.region_name,
                    "hemisphere": region.hemisphere,
                    "parent_region_id": region.parent_region_id,
                    "hierarchy": region.hierarchy,
                    "raw_contents_for_export": raw_summary,
                }
            )
    return rows


def export_omit_outputs(dataset: OverlayDataset, session: OmitSessionState, output_dir: Path) -> dict[str, Path]:
    """Write omit CSVs, session YAML, QC JPG images, and omitted quantification tables."""

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows = build_omit_rows(dataset, session)
    csv_path = output_dir / "omit_regions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    session_path = output_dir / "omit_session.yaml"
    save_session(session, session_path)

    raw_contents = [content for content in session.visible_contents if content.endswith("_raw_image")]
    if not raw_contents:
        raw_contents = dataset.raw_contents
    outline_contents = [info.content for info in dataset.channels if info.is_outline]
    image_contents = [*raw_contents, *outline_contents]

    exported_paths: list[Path] = []
    for slice_info in dataset.slices:
        stack = load_overlay_stack(slice_info)
        image = render_qc_image(
            stack=stack,
            channels=dataset.channels,
            visible_contents=image_contents,
            omitted_regions=session.omitted_regions_by_slice.get(slice_info.key, []),
        )
        jpg_path = images_dir / f"{slice_info.animal_id}_{slice_info.section_id}_omit_qc.jpg"
        image.save(jpg_path, format="JPEG", quality=92, optimize=True)
        exported_paths.append(jpg_path)

    table_outputs = _export_omitted_quantification_tables(dataset, session, output_dir)
    return {
        "csv": csv_path,
        "session": session_path,
        "images_dir": images_dir,
        "first_image": exported_paths[0] if exported_paths else images_dir,
        **table_outputs,
    }


def region_info_for_code(region_lookup: dict[int, RegionInfo], display_code: int) -> RegionInfo:
    """Return region metadata for one display code, with sign-based fallback decoding."""

    code = int(display_code)
    if code in region_lookup:
        return region_lookup[code]
    hemisphere = "left" if code < 0 else "right"
    region_id = abs(code)
    return RegionInfo(
        display_code=code,
        region_id=region_id,
        region_name=f"Region {region_id}",
        hemisphere=hemisphere,
    )


def _export_omitted_quantification_tables(
    dataset: OverlayDataset,
    session: OmitSessionState,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate omitted cell/region/section tables from existing QDF2 outputs."""

    results_dir = dataset.overlay_dir.parent
    cell_path = results_dir / "cell_level.csv"
    region_path = results_dir / "region_summary.csv"
    section_path = results_dir / "section_summary.csv"
    section_channel_path = results_dir / "section_channel_summary.csv"
    section_input_path = section_channel_path if section_channel_path.exists() else section_path
    if not cell_path.exists() or not region_path.exists() or not section_input_path.exists():
        return {}

    base_cell = pd.read_csv(cell_path)
    base_region = pd.read_csv(region_path)
    base_section = pd.read_csv(section_input_path)
    if "pixel_area_um2" not in base_section.columns:
        return {}

    augmented_cell = _augment_cell_level_with_patch_info(base_cell, dataset)
    filtered_cell, exact_overlap = _apply_omit_to_cell_level(augmented_cell, dataset, session)
    section_summary_omit = _build_section_summary_omit(filtered_cell, base_section)
    region_summary_omit = _build_region_summary_omit(
        filtered_cell,
        base_region,
        section_summary_omit,
        dataset,
        session,
        exact_overlap=exact_overlap,
    )

    cell_out = output_dir / "cell_level_omit.csv"
    region_out = output_dir / "region_summary_omit.csv"
    section_out = output_dir / "section_summary_omit.csv"
    section_channel_out = output_dir / "section_channel_summary_omit.csv"
    section_region_omit = build_section_region_summary(region_summary_omit, section_summary_omit)
    filtered_cell.to_csv(cell_out, index=False, encoding="utf-8-sig")
    region_summary_omit.to_csv(region_out, index=False, encoding="utf-8-sig")
    section_region_omit.to_csv(section_out, index=False, encoding="utf-8-sig")
    section_summary_omit.to_csv(section_channel_out, index=False, encoding="utf-8-sig")
    return {
        "cell_level_omit": cell_out,
        "region_summary_omit": region_out,
        "section_summary_omit": section_out,
        "section_channel_summary_omit": section_channel_out,
    }


def _augment_cell_level_with_patch_info(cell_df: pd.DataFrame, dataset: OverlayDataset) -> pd.DataFrame:
    """Ensure cell-level rows have atlas patch identifiers, even for legacy QDF2 outputs."""

    working = cell_df.copy()
    for column in ("atlas_display_code", "atlas_patch_component", "atlas_patch_id"):
        if column not in working.columns:
            working[column] = 0 if column != "atlas_patch_id" else ""

    working["atlas_display_code"] = pd.to_numeric(working["atlas_display_code"], errors="coerce").fillna(0).astype(int)
    working["atlas_patch_component"] = pd.to_numeric(working["atlas_patch_component"], errors="coerce").fillna(0).astype(int)
    working["atlas_patch_id"] = working["atlas_patch_id"].fillna("").astype(str)

    patch_ready = (
        working["atlas_patch_id"].str.strip().ne("")
        & working["atlas_patch_component"].gt(0)
        & working["atlas_display_code"].ne(0)
    )
    if bool(patch_ready.all()):
        return working

    slice_lookup = {(slice_info.animal_id, slice_info.section_id): slice_info for slice_info in dataset.slices}
    section_only_lookup = {slice_info.section_id: slice_info for slice_info in dataset.slices}
    context_cache: dict[str, dict[str, Any]] = {}

    working["animal_id"] = working["animal_id"].astype(str)
    working["section_id"] = working["section_id"].astype(str)
    working["hemisphere"] = working["hemisphere"].astype(str).str.lower()
    working["region_id"] = pd.to_numeric(working["region_id"], errors="coerce").fillna(0).astype(int)
    working["centroid_x_px"] = pd.to_numeric(working["centroid_x_px"], errors="coerce")
    working["centroid_y_px"] = pd.to_numeric(working["centroid_y_px"], errors="coerce")

    for (animal_id, section_id), index_labels in working.groupby(["animal_id", "section_id"], sort=False).groups.items():
        idx = pd.Index(index_labels)
        incomplete_mask = ~patch_ready.loc[idx]
        if not bool(incomplete_mask.any()):
            continue

        slice_info = slice_lookup.get((str(animal_id), str(section_id))) or section_only_lookup.get(str(section_id))
        if slice_info is None:
            continue

        context = _slice_context(dataset, slice_info, context_cache)
        atlas_plane = context["atlas_plane"]
        slice_frame = working.loc[idx]

        x = np.rint(slice_frame["centroid_x_px"].fillna(-1).to_numpy(dtype=np.float64)).astype(np.int32)
        y = np.rint(slice_frame["centroid_y_px"].fillna(-1).to_numpy(dtype=np.float64)).astype(np.int32)
        valid_xy = (x >= 0) & (x < atlas_plane.shape[1]) & (y >= 0) & (y < atlas_plane.shape[0])

        display_codes = slice_frame["atlas_display_code"].to_numpy(dtype=np.int32, copy=True)
        need_sample = (display_codes == 0) & valid_xy
        if np.any(need_sample):
            display_codes[need_sample] = atlas_plane[y[need_sample], x[need_sample]].astype(np.int32, copy=False)

        region_ids = slice_frame["region_id"].to_numpy(dtype=np.int32, copy=False)
        hemispheres = slice_frame["hemisphere"].to_numpy(dtype=object, copy=False)
        need_fallback = display_codes == 0
        if np.any(need_fallback):
            left_mask = need_fallback & (region_ids > 0) & (hemispheres == "left")
            right_mask = need_fallback & (region_ids > 0) & (hemispheres == "right")
            display_codes[left_mask] = -region_ids[left_mask]
            display_codes[right_mask] = region_ids[right_mask]

        component_labels = np.zeros(len(slice_frame), dtype=np.int32)
        for display_code in np.unique(display_codes):
            if int(display_code) == 0:
                continue
            code_mask = (display_codes == int(display_code)) & valid_xy
            if not np.any(code_mask):
                continue
            component_map = _component_map_for_display_code(context, int(display_code))
            component_labels[code_mask] = component_map[y[code_mask], x[code_mask]].astype(np.int32, copy=False)

        valid_component = component_labels > 0
        working.loc[idx, "atlas_display_code"] = display_codes
        working.loc[idx, "atlas_patch_component"] = component_labels
        patch_ids = np.full(len(slice_frame), "", dtype=object)
        if np.any(valid_component):
            patch_ids[valid_component] = [
                _format_patch_id(str(animal_id), str(section_id), int(code), int(component))
                for code, component in zip(display_codes[valid_component], component_labels[valid_component], strict=False)
            ]
        working.loc[idx, "atlas_patch_id"] = patch_ids

    working["atlas_patch_id"] = working["atlas_patch_id"].fillna("").astype(str)

    return working


def _apply_omit_to_cell_level(
    cell_df: pd.DataFrame,
    dataset: OverlayDataset,
    session: OmitSessionState,
) -> tuple[pd.DataFrame, bool]:
    """Drop omitted atlas patches and refresh overlap annotations when possible."""

    omit_rows = build_omit_rows(dataset, session)
    omit_patch_ids = {str(row["atlas_patch_id"]) for row in omit_rows}
    if omit_patch_ids:
        omit_mask = cell_df["atlas_patch_id"].astype(str).isin(omit_patch_ids)
        affected_group_ids = set(
            cell_df.loc[omit_mask, "overlap_group_id"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().tolist()
        )
        filtered = cell_df.loc[~omit_mask].copy()
    else:
        affected_group_ids = set()
        filtered = cell_df.copy()

    exact_overlap = (
        "overlap_group_id" in filtered.columns
        and filtered["overlap_group_id"].astype(str).str.strip().ne("").any()
    )
    if exact_overlap and affected_group_ids:
        filtered = _recompute_overlap_membership(filtered, affected_group_ids=affected_group_ids)
    return filtered, bool(exact_overlap)


def _recompute_overlap_membership(
    cell_df: pd.DataFrame,
    *,
    affected_group_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Refresh overlap flags/labels after omitted patches removed some group members."""

    working = cell_df.copy()
    channel_columns = _channel_columns(working)
    if not channel_columns or "overlap_group_id" not in working.columns:
        return working

    for channel in channel_columns:
        working[channel] = pd.to_numeric(working[channel], errors="coerce").fillna(0).astype(int)

    group_ids = working["overlap_group_id"].astype(str).str.strip()
    nonempty = group_ids.ne("")
    if affected_group_ids is None:
        target_mask = nonempty
    else:
        if not affected_group_ids:
            return working
        target_mask = nonempty & group_ids.isin(affected_group_ids)

    for overlap_group_id, group_frame in working.loc[target_mask].groupby("overlap_group_id", sort=False):
        indices = list(group_frame.index)
        channels_present = sorted(
            {str(value).upper() for value in group_frame["image_channel"].tolist() if str(value).strip()},
            key=str.upper,
        )
        if not channels_present:
            continue
        for channel in channel_columns:
            working.loc[indices, channel] = 1 if channel.upper() in channels_present else 0
        working.loc[indices, "overlap_group"] = _group_label_from_channels(channels_present)

    if affected_group_ids is None:
        singles = working.loc[~nonempty, "image_channel"].fillna("").astype(str).str.upper()
        if not singles.empty:
            for channel in channel_columns:
                working.loc[~nonempty, channel] = (singles == channel.upper()).astype(int).to_numpy()
            labels = singles.map(lambda channel_name: _group_label_from_channels([channel_name]) if channel_name else "")
            working.loc[~nonempty, "overlap_group"] = labels.to_numpy()
    return working


def _build_section_summary_omit(cell_df: pd.DataFrame, base_section: pd.DataFrame) -> pd.DataFrame:
    """Recalculate section-level summaries after omitted patches removed cells."""

    rows: list[dict[str, Any]] = []
    for record in base_section.to_dict(orient="records"):
        animal_id = str(record.get("animal_id", ""))
        section_id = str(record.get("section_id", ""))
        image_channel = str(record.get("image_channel", ""))
        mask = (
            cell_df["animal_id"].astype(str) == animal_id
        ) & (
            cell_df["section_id"].astype(str) == section_id
        ) & (
            cell_df["image_channel"].astype(str) == image_channel
        )
        subset = cell_df.loc[mask]
        row = dict(record)
        row["n_detected_cells"] = int(len(subset))
        row["n_unassigned_cells"] = int((subset["region_id"].fillna(0).astype(int) == 0).sum()) if not subset.empty else 0
        row["left_count"] = int((subset["hemisphere"].astype(str).str.lower() == "left").sum()) if not subset.empty else 0
        row["right_count"] = int((subset["hemisphere"].astype(str).str.lower() == "right").sum()) if not subset.empty else 0
        rows.append(row)
    output = pd.DataFrame(rows)
    if not output.empty:
        output = output.reindex(columns=list(base_section.columns))
    return output


def _build_region_summary_omit(
    cell_df: pd.DataFrame,
    base_region: pd.DataFrame,
    section_summary: pd.DataFrame,
    dataset: OverlayDataset,
    session: OmitSessionState,
    *,
    exact_overlap: bool,
) -> pd.DataFrame:
    """Recalculate region summaries from filtered cells and remaining atlas areas."""

    if cell_df.empty:
        return pd.DataFrame(columns=list(base_region.columns))

    base_columns = list(base_region.columns)
    overlap_labels = sorted({column[:-8] for column in base_columns if column.endswith("_n_cells")})
    overlap_defaults: dict[str, object] = {}
    for label in overlap_labels:
        overlap_defaults[f"{label}_n_cells"] = 0
        overlap_defaults[f"{label}_mean_integrated_intensity"] = None
        overlap_defaults[f"{label}_mean_cell_area"] = None

    metadata_by_channel: dict[tuple[str, str, str], dict[str, object]] = {}
    for record in base_region.to_dict(orient="records"):
        key = (
            str(record.get("animal_id", "")),
            str(record.get("section_id", "")),
            str(record.get("image_channel", "")),
        )
        metadata_by_channel.setdefault(key, dict(record))

    pixel_area_lookup = (
        section_summary.groupby(["animal_id", "section_id"], as_index=False)["pixel_area_um2"].first()
        .set_index(["animal_id", "section_id"])["pixel_area_um2"]
        .to_dict()
    )
    area_lookup_by_section = _remaining_region_area_lookup(dataset, session, pixel_area_lookup)

    rows: list[dict[str, Any]] = []
    for (animal_id, section_id), section_frame in cell_df.groupby(["animal_id", "section_id"], sort=True):
        overlap_metrics = _compute_overlap_metrics(section_frame, overlap_labels, exact_overlap=exact_overlap)
        area_lookup = area_lookup_by_section.get((str(animal_id), str(section_id)), {})

        for image_channel, channel_frame in section_frame.groupby("image_channel", sort=True):
            metadata = metadata_by_channel.get((str(animal_id), str(section_id), str(image_channel)), {})
            grouped: dict[tuple[int, str], pd.DataFrame] = {}
            for (region_id, hemisphere), region_frame in channel_frame.groupby(["region_id", "hemisphere"], sort=True):
                region_id_int = int(region_id)
                if region_id_int == 0:
                    continue
                grouped[(region_id_int, str(hemisphere))] = region_frame
            for region_id, region_frame in channel_frame.groupby("region_id", sort=True):
                region_id_int = int(region_id)
                if region_id_int == 0:
                    continue
                grouped[(region_id_int, "total")] = region_frame

            for (region_id, hemisphere), region_frame in sorted(grouped.items()):
                total_integrated = float(region_frame["integrated_intensity"].sum())
                mean_intensity = float(region_frame["mean_intensity"].mean()) if not region_frame.empty else 0.0
                mean_area = float(region_frame["area_px"].mean()) if not region_frame.empty else 0.0
                patch_summary = _region_patch_summary_from_frame(region_frame)
                area_um2 = float(area_lookup.get((region_id, hemisphere), 0.0))
                density = float(len(region_frame) / (area_um2 / 1_000_000.0)) if area_um2 > 0 else None
                row = {
                    "animal_id": animal_id,
                    "section_id": section_id,
                    "channel": metadata.get("channel", image_channel),
                    "image_channel": image_channel,
                    "channel_or_combination": metadata.get("channel_or_combination", image_channel),
                    "region_id": region_id,
                    "region_name": str(region_frame["region_name"].iloc[0]),
                    "hemisphere": hemisphere,
                    "n_cells": int(len(region_frame)),
                    "total_integrated_intensity": total_integrated,
                    "mean_integrated_intensity": mean_intensity,
                    "mean_cell_area": mean_area,
                    "density_if_possible": density,
                    "region_area_um2": area_um2,
                    "atlas_patch_count": patch_summary["atlas_patch_count"],
                    "atlas_display_codes": patch_summary["atlas_display_codes"],
                    "atlas_patch_components": patch_summary["atlas_patch_components"],
                    "atlas_patch_ids": patch_summary["atlas_patch_ids"],
                }
                if "summary_source" in base_columns:
                    row["summary_source"] = metadata.get("summary_source", "omitByQDF3")
                row.update(overlap_defaults)
                row.update(overlap_metrics.get((region_id, hemisphere), {}))
                rows.append(row)

    output = pd.DataFrame(rows)
    if output.empty:
        return pd.DataFrame(columns=base_columns)
    for column in base_columns:
        if column not in output.columns:
            output[column] = pd.NA
    return output.reindex(columns=base_columns).sort_values(
        ["animal_id", "section_id", "image_channel", "region_id", "hemisphere"]
    ).reset_index(drop=True)


def _remaining_region_area_lookup(
    dataset: OverlayDataset,
    session: OmitSessionState,
    pixel_area_lookup: dict[tuple[str, str], float],
) -> dict[tuple[str, str], dict[tuple[int, str], float]]:
    """Return remaining atlas area per section/region after omitted patches removed."""

    cache: dict[str, dict[str, Any]] = {}
    area_lookup_by_section: dict[tuple[str, str], dict[tuple[int, str], float]] = {}
    for slice_info in dataset.slices:
        pixel_area_um2 = float(pixel_area_lookup.get((slice_info.animal_id, slice_info.section_id), 0.0))
        context = _slice_context(dataset, slice_info, cache)
        atlas_plane = context["atlas_plane"]
        valid_mask = atlas_plane != 0
        for selection in session.omitted_regions_by_slice.get(slice_info.key, []):
            component_map = _component_map_for_display_code(context, selection.display_code)
            valid_mask &= ~(component_map == int(selection.component_label))
        lookup: dict[tuple[int, str], float] = {}
        remaining_codes = atlas_plane[valid_mask]
        if remaining_codes.size:
            region_ids, total_counts = np.unique(np.abs(remaining_codes), return_counts=True)
            for region_id, count in zip(region_ids, total_counts, strict=False):
                lookup[(int(region_id), "total")] = float(count) * pixel_area_um2
            for hemisphere_name, sign in (("left", -1), ("right", 1)):
                hemi_codes = remaining_codes[remaining_codes * sign > 0]
                if hemi_codes.size:
                    region_ids, hemi_counts = np.unique(np.abs(hemi_codes), return_counts=True)
                    for region_id, count in zip(region_ids, hemi_counts, strict=False):
                        lookup[(int(region_id), hemisphere_name)] = float(count) * pixel_area_um2
        area_lookup_by_section[(slice_info.animal_id, slice_info.section_id)] = lookup
    return area_lookup_by_section


def _region_patch_summary_from_frame(region_frame: pd.DataFrame) -> dict[str, object]:
    """Summarize atlas patch membership for one omitted region-summary row."""

    if region_frame.empty:
        return {
            "atlas_patch_count": 0,
            "atlas_display_codes": "",
            "atlas_patch_components": "",
            "atlas_patch_ids": "",
        }

    display_codes = (
        pd.to_numeric(region_frame.get("atlas_display_code", pd.Series(dtype=int)), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    patch_components = (
        pd.to_numeric(region_frame.get("atlas_patch_component", pd.Series(dtype=int)), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    patch_ids = region_frame.get("atlas_patch_id", pd.Series(dtype=str)).fillna("").astype(str).str.strip()

    valid_display_codes = sorted({int(code) for code in display_codes.tolist() if int(code) != 0})
    valid_components = sorted(
        {
            f"{int(code)}:{int(component)}"
            for code, component in zip(display_codes.tolist(), patch_components.tolist(), strict=False)
            if int(code) != 0 and int(component) > 0
        }
    )
    valid_patch_ids = sorted({value for value in patch_ids.tolist() if value})
    return {
        "atlas_patch_count": int(len(valid_patch_ids)),
        "atlas_display_codes": ";".join(str(code) for code in valid_display_codes),
        "atlas_patch_components": ";".join(valid_components),
        "atlas_patch_ids": ";".join(valid_patch_ids),
    }


def _compute_overlap_metrics(
    section_frame: pd.DataFrame,
    overlap_labels: list[str],
    *,
    exact_overlap: bool,
) -> dict[tuple[int, str], dict[str, object]]:
    """Compute overlap metrics for one section, exactly when overlap_group_id exists."""

    if not overlap_labels:
        return {}
    channel_columns = _channel_columns(section_frame)
    metrics_by_key: dict[tuple[int, str], dict[str, object]] = {}

    if exact_overlap and "overlap_group_id" in section_frame.columns:
        groups = _exact_overlap_groups_frame(section_frame, channel_columns)
        for label in overlap_labels:
            spec = parse_overlap_spec(label)
            matched_groups = groups.loc[groups["region_id"].gt(0)].copy()
            for channel, is_positive in spec.terms:
                expected = 1 if is_positive else 0
                if channel.upper() in matched_groups.columns:
                    matched_groups = matched_groups.loc[matched_groups[channel.upper()].eq(expected)]
                elif is_positive:
                    matched_groups = matched_groups.iloc[0:0]
                    break
            if matched_groups.empty:
                continue

            total_grouped = matched_groups.groupby("region_id", sort=True).agg(
                n_cells=("region_id", "size"),
                mean_integrated_intensity=("mean_intensity", "mean"),
                mean_cell_area=("mean_cell_area", "mean"),
            )
            for region_id, record in total_grouped.iterrows():
                metrics = metrics_by_key.setdefault((int(region_id), "total"), {})
                metrics[f"{label}_n_cells"] = int(record["n_cells"])
                metrics[f"{label}_mean_integrated_intensity"] = float(record["mean_integrated_intensity"])
                metrics[f"{label}_mean_cell_area"] = float(record["mean_cell_area"])

            hemi_grouped = matched_groups.loc[matched_groups["hemisphere"].isin(["left", "right"])].groupby(
                ["region_id", "hemisphere"],
                sort=True,
            ).agg(
                n_cells=("region_id", "size"),
                mean_integrated_intensity=("mean_intensity", "mean"),
                mean_cell_area=("mean_cell_area", "mean"),
            )
            for (region_id, hemisphere), record in hemi_grouped.iterrows():
                metrics = metrics_by_key.setdefault((int(region_id), str(hemisphere)), {})
                metrics[f"{label}_n_cells"] = int(record["n_cells"])
                metrics[f"{label}_mean_integrated_intensity"] = float(record["mean_integrated_intensity"])
                metrics[f"{label}_mean_cell_area"] = float(record["mean_cell_area"])
        return metrics_by_key

    channel_series = {
        channel.upper(): pd.to_numeric(section_frame[channel], errors="coerce").fillna(0).astype(int)
        for channel in channel_columns
    }
    zero_series = pd.Series(0, index=section_frame.index, dtype=int)

    for label in overlap_labels:
        spec = parse_overlap_spec(label)
        positive_count = max(1, len(spec.positive_channels))
        mask = pd.Series(True, index=section_frame.index, dtype=bool)
        for channel, is_positive in spec.terms:
            flags = channel_series.get(channel.upper(), zero_series)
            mask &= flags.eq(1 if is_positive else 0)
        matched_rows = section_frame.loc[mask & (section_frame["region_id"].fillna(0).astype(int) > 0)]
        grouped_matches: dict[tuple[int, str], pd.DataFrame] = {}
        for (region_id, hemisphere), region_frame in matched_rows.groupby(["region_id", "hemisphere"], sort=True):
            grouped_matches[(int(region_id), str(hemisphere))] = region_frame
        for region_id, region_frame in matched_rows.groupby("region_id", sort=True):
            grouped_matches[(int(region_id), "total")] = region_frame
        for key, region_frame in grouped_matches.items():
            metrics = metrics_by_key.setdefault(key, {})
            metrics[f"{label}_n_cells"] = int(round(len(region_frame) / positive_count))
            metrics[f"{label}_mean_integrated_intensity"] = float(region_frame["mean_intensity"].mean()) if not region_frame.empty else None
            metrics[f"{label}_mean_cell_area"] = float(region_frame["area_px"].mean()) if not region_frame.empty else None
    return metrics_by_key


def _exact_overlap_groups_frame(section_frame: pd.DataFrame, channel_columns: list[str]) -> pd.DataFrame:
    """Collapse remaining rows into exact overlap groups using overlap_group_id."""

    nonempty = section_frame["overlap_group_id"].astype(str).str.strip().ne("")
    grouped = section_frame.loc[nonempty].copy()
    if grouped.empty:
        columns = ["overlap_group_id", "region_id", "hemisphere", "mean_intensity", "mean_cell_area", *channel_columns]
        return pd.DataFrame(columns=columns)

    grouped["region_id"] = pd.to_numeric(grouped["region_id"], errors="coerce").fillna(0).astype(int)
    grouped["hemisphere"] = grouped["hemisphere"].fillna("").astype(str).str.lower()

    base = grouped.groupby("overlap_group_id", sort=False).agg(
        mean_intensity=("mean_intensity", "mean"),
        mean_cell_area=("area_px", "mean"),
        **{channel.upper(): (channel, "max") for channel in channel_columns},
    )

    positive_region = (
        grouped.loc[grouped["region_id"] > 0, ["overlap_group_id", "region_id"]]
        .drop_duplicates(["overlap_group_id", "region_id"])
        .groupby("overlap_group_id", sort=False)["region_id"]
        .first()
    )
    hemisphere = (
        grouped.loc[grouped["hemisphere"].isin(["left", "right"]), ["overlap_group_id", "hemisphere"]]
        .drop_duplicates(["overlap_group_id", "hemisphere"])
        .groupby("overlap_group_id", sort=False)["hemisphere"]
        .first()
    )

    base["region_id"] = positive_region.reindex(base.index).fillna(0).astype(int)
    base["hemisphere"] = hemisphere.reindex(base.index).fillna("unknown").astype(str)
    return base.reset_index()


def _row_matches_overlap_spec(
    row: pd.Series,
    spec_terms: tuple[tuple[str, bool], ...],
    channel_columns: list[str],
) -> bool:
    for channel, is_positive in spec_terms:
        if channel.upper() not in channel_columns:
            flag = 0
        else:
            value = row.get(channel.upper(), 0)
            flag = 0 if pd.isna(value) else int(value)
        if is_positive and flag != 1:
            return False
        if not is_positive and flag != 0:
            return False
    return True


def _channel_columns(frame: pd.DataFrame) -> list[str]:
    return sorted([column for column in frame.columns if CHANNEL_COLUMN_PATTERN.fullmatch(str(column).upper())], key=str.upper)


def _group_label_from_channels(channels: list[str]) -> str:
    prefixes = {1: "single", 2: "double", 3: "triple", 4: "quadruple"}
    ordered = sorted({str(channel).upper() for channel in channels if str(channel).strip()}, key=str.upper)
    return f"{prefixes.get(len(ordered), f'{len(ordered)}ch')}_" + "_".join(ordered)


def _slice_context(
    dataset: OverlayDataset,
    slice_info: OverlaySliceInfo,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Load and cache per-slice atlas planes/components."""

    cached = cache.get(slice_info.key)
    if cached is not None:
        return cached
    stack = load_overlay_stack(slice_info)
    atlas_infos = [info for info in dataset.channels if info.is_display_code]
    if not atlas_infos:
        raise KeyError("atlas_display_code plane was not found in the channel map.")
    atlas_plane = np.asarray(stack[atlas_infos[0].plane_index], dtype=np.int64)
    cached = {"atlas_plane": atlas_plane, "component_maps": {}}
    cache[slice_info.key] = cached
    return cached


def _component_map_for_display_code(context: dict[str, Any], display_code: int) -> np.ndarray:
    """Return cached component labels for one display code in one slice."""

    cache: dict[int, np.ndarray] = context["component_maps"]
    if int(display_code) not in cache:
        cache[int(display_code)] = component_label_map_for_code(context["atlas_plane"], int(display_code))
    return cache[int(display_code)]


def _format_patch_id(animal_id: str, section_id: str, display_code: int, component_label: int) -> str:
    return f"{animal_id}::{section_id}::{int(display_code)}::{int(component_label)}"


def _single_path_or_raise(directory: Path, pattern: str, label: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No {label} matching {pattern!r} found in {directory}")
    return matches[0]


def _find_codebook(overlay_dir: Path) -> Path | None:
    candidates = [
        overlay_dir.parent / "atlas_display_codebook.csv",
        overlay_dir / "atlas_display_codebook.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_channel_map(channel_map_path: Path) -> list[OverlayChannelInfo]:
    frame = pd.read_excel(channel_map_path, sheet_name="overlay_stack")
    channels: list[OverlayChannelInfo] = []
    for row in frame.to_dict(orient="records"):
        channels.append(
            OverlayChannelInfo(
                channel_index=int(row["channel_index"]),
                channel_name=str(row.get("channel_name", f"CH{row['channel_index']}")),
                content=str(row.get("content", "")),
                source_channel="" if pd.isna(row.get("source_channel")) else str(row.get("source_channel", "")),
                notes="" if pd.isna(row.get("notes")) else str(row.get("notes", "")),
            )
        )
    channels.sort(key=lambda info: info.channel_index)
    return channels


def _load_region_lookup(codebook_path: Path | None) -> dict[int, RegionInfo]:
    if codebook_path is None or not codebook_path.exists():
        return {}
    frame = pd.read_csv(codebook_path)
    lookup: dict[int, RegionInfo] = {}
    for row in frame.to_dict(orient="records"):
        lookup[int(row["display_code"])] = RegionInfo(
            display_code=int(row["display_code"]),
            region_id=int(row["region_id"]),
            region_name=str(row["region_name"]),
            hemisphere=str(row.get("hemisphere", "unknown")),
            parent_region_id=None if pd.isna(row.get("parent_region_id")) else int(row["parent_region_id"]),
            hierarchy="" if pd.isna(row.get("hierarchy")) else str(row.get("hierarchy", "")),
        )
    return lookup


def _parse_overlay_slice(path: Path) -> OverlaySliceInfo:
    match = OVERLAY_PATTERN.match(path.name)
    if match:
        animal_id = match.group("animal").strip()
        section_id = match.group("section").strip()
    else:
        animal_id = path.stem
        section_id = path.stem
    return OverlaySliceInfo(
        overlay_path=path,
        animal_id=animal_id,
        section_id=section_id,
        display_name=f"{animal_id} | {section_id}",
    )


def _overlay_sort_key(name: str) -> tuple[str, int, str]:
    match = OVERLAY_PATTERN.match(name)
    if not match:
        return (name, 0, name)
    animal = match.group("animal")
    section = match.group("section")
    number_match = re.search(r"(\d+)$", section)
    section_number = int(number_match.group(1)) if number_match else 0
    return (animal, section_number, section)
