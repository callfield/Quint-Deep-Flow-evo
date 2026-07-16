"""QDFevo_2_AtlasFitter omit-state import helpers for QDF2 quantification."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage
import tifffile

from io_utils.discovery import extract_section_id
from quality_check.core import (
    _channel_columns,
    _component_map_for_display_code,
    _compute_overlap_metrics,
    _recompute_overlap_membership,
    _region_patch_summary_from_frame,
    _slice_context,
    build_omit_rows,
    create_default_session,
    load_overlay_dataset,
    save_session,
)
from quality_check.models import OmitRegionSelection, OmitSessionState, OverlayDataset
from registration.parser import parse_registration_file


LOG = logging.getLogger(__name__)
OMIT_LAYER_CONTENT = "omit_region_mask"
OMIT_PLANE_VALUE = 65535


def _overlay_stack_shape(path: Path) -> tuple[int, int, int]:
    """Return overlay stack shape as (planes, height, width) without reading pixel data."""

    with tifffile.TiffFile(path) as tif:
        shape = tuple(int(value) for value in tif.series[0].shape)
    if len(shape) == 2:
        return (1, int(shape[0]), int(shape[1]))
    if len(shape) == 3:
        return (int(shape[0]), int(shape[1]), int(shape[2]))
    raise ValueError(f"Expected 2D/3D overlay stack in {path}, got shape {shape}")


def _read_overlay_plane(path: Path, plane_index: int) -> np.ndarray:
    """Read one overlay plane, falling back to full-stack indexing for nonstandard TIFFs."""

    try:
        plane = tifffile.imread(path, key=int(plane_index))
        if plane.ndim == 2:
            return np.asarray(plane)
    except Exception:
        pass
    stack = tifffile.imread(path)
    if stack.ndim == 2:
        if int(plane_index) != 0:
            raise IndexError(f"Overlay plane {plane_index} out of range for {path}")
        return np.asarray(stack)
    return np.asarray(stack[int(plane_index)])


def find_qdf1_evo_omit_state_path(registration_json_path: Path) -> Path | None:
    """Return the omit-state JSON next to a QDFevo_2_AtlasFitter registration JSON when present."""

    registration_json_path = Path(registration_json_path).expanduser().resolve()
    candidate = registration_json_path.with_name(f"{registration_json_path.stem}_omit_state.json")
    return candidate if candidate.exists() else None


def apply_qdf1_evo_omit_to_results(
    *,
    output_dir: Path,
    registration_json_path: Path,
    logger: logging.Logger | None = None,
) -> dict[str, object]:
    """Import QDFevo_2_AtlasFitter omit strokes, append overlay plane, and overwrite summary tables."""

    logger = logger or LOG
    output_dir = Path(output_dir).expanduser().resolve()
    registration_json_path = Path(registration_json_path).expanduser().resolve()
    omit_state_path = find_qdf1_evo_omit_state_path(registration_json_path)
    if omit_state_path is None:
        return {"enabled": False, "reason": "omit_state_missing"}

    overlay_dir = output_dir / "overlay"
    if not overlay_dir.exists():
        return {"enabled": False, "reason": "overlay_dir_missing", "omit_state_path": str(omit_state_path)}

    logger.info("Importing QDFevo_2_AtlasFitter omit state: %s", omit_state_path)
    dataset = load_overlay_dataset(overlay_dir)
    omit_masks_by_slice = _omit_masks_from_qdf1_evo_state(dataset, omit_state_path, registration_json_path)
    session = _session_from_omit_masks(dataset, omit_masks_by_slice)
    omit_rows = build_omit_rows(dataset, session)
    overlay_update = _write_omit_plane_to_overlay(dataset, omit_masks_by_slice)
    table_update = _overwrite_tables_with_omit(output_dir, dataset, omit_masks_by_slice)

    report_path = output_dir / "qdf1_atlasfitter_imported_omit_regions.csv"
    pd.DataFrame(omit_rows).to_csv(report_path, index=False)
    session_path = output_dir / "qdf1_atlasfitter_imported_omit_session.yaml"
    save_session(session, session_path)

    logger.info(
        "Applied QDFevo_2_AtlasFitter omit state: %s touched atlas patches across %s slices",
        len(omit_rows),
        sum(1 for values in session.omitted_regions_by_slice.values() if values),
    )
    return {
        "enabled": True,
        "omit_state_path": str(omit_state_path),
        "omit_report_csv": str(report_path),
        "omit_session_yaml": str(session_path),
        "omitted_patch_count": int(len(omit_rows)),
        "slices_with_omit": int(sum(1 for values in session.omitted_regions_by_slice.values() if values)),
        **overlay_update,
        **table_update,
    }


def _normalize_filename_key(name: str) -> str:
    return Path(str(name)).name.strip().lower().replace(" ", "_")


def _registration_image_size_by_name(registration_json_path: Path) -> dict[str, tuple[int, int]]:
    registration = parse_registration_file(registration_json_path)
    sizes: dict[str, tuple[int, int]] = {}
    for registration_slice in registration.slices:
        try:
            width = int(registration_slice.width)
            height = int(registration_slice.height)
        except (TypeError, ValueError):
            continue
        sizes[_normalize_filename_key(registration_slice.filename)] = (width, height)
    return sizes


def _omit_masks_from_qdf1_evo_state(
    dataset: OverlayDataset,
    omit_state_path: Path,
    registration_json_path: Path,
) -> dict[str, np.ndarray]:
    """Convert QDFevo_2_AtlasFitter omit strokes into per-slice overlay-space masks."""

    with Path(omit_state_path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle) or {}

    registration_sizes = _registration_image_size_by_name(registration_json_path)
    slice_entries = raw.get("slices", {}) if isinstance(raw, dict) else {}
    section_to_entries: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for image_name, payload in slice_entries.items():
        if not isinstance(payload, dict):
            continue
        section_id = extract_section_id(str(image_name))
        section_to_entries.setdefault(section_id, []).append((str(image_name), payload))

    omit_masks_by_slice: dict[str, np.ndarray] = {}
    for slice_info in dataset.slices:
        entries = section_to_entries.get(slice_info.section_id, [])
        if not entries:
            continue
        _, height, width = _overlay_stack_shape(slice_info.overlay_path)
        combined_mask = np.zeros((height, width), dtype=bool)
        for image_name, payload in entries:
            strokes = payload.get("omit_strokes", [])
            source_name = payload.get("image_name", "") or payload.get("filename", "") or image_name
            source_size = registration_sizes.get(_normalize_filename_key(source_name))
            mask_file_mask = _omit_mask_file_mask(
                width=width,
                height=height,
                payload=payload,
                omit_state_path=omit_state_path,
            )
            if mask_file_mask is not None and np.any(mask_file_mask):
                combined_mask |= mask_file_mask
            for stroke in strokes if isinstance(strokes, list) else []:
                stroke_mask = _stroke_mask(width=width, height=height, stroke=stroke, source_size=source_size)
                if stroke_mask is None or not np.any(stroke_mask):
                    continue
                combined_mask |= stroke_mask
        omit_masks_by_slice[slice_info.key] = combined_mask
    return omit_masks_by_slice


def _resolve_omit_mask_file(payload: dict[str, Any], omit_state_path: Path) -> Path | None:
    raw_mask_file = str(payload.get("mask_file", "") or "").strip()
    if not raw_mask_file:
        return None
    raw_path = Path(raw_mask_file)
    if raw_path.is_absolute():
        return raw_path if raw_path.exists() else None
    candidates = [omit_state_path.parent / raw_path]
    stem = omit_state_path.stem
    if stem.endswith("_omit_state"):
        candidates.append(omit_state_path.with_name(f"{stem[: -len('_omit_state')]}_omitMasks") / raw_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _omit_mask_file_mask(
    *,
    width: int,
    height: int,
    payload: dict[str, Any],
    omit_state_path: Path,
) -> np.ndarray | None:
    mask_path = _resolve_omit_mask_file(payload, omit_state_path)
    if mask_path is None:
        return None
    with Image.open(mask_path) as image:
        mask_image = image.convert("L")
        if mask_image.size != (int(width), int(height)):
            mask_image = mask_image.resize((int(width), int(height)), resample=Image.Resampling.NEAREST)
        return np.asarray(mask_image, dtype=np.uint8) > 0


def _session_from_omit_masks(
    dataset: OverlayDataset,
    omit_masks_by_slice: dict[str, np.ndarray],
) -> OmitSessionState:
    """Create a touched-patch session summary from imported omit masks."""

    session = create_default_session(dataset)
    context_cache: dict[str, dict[str, Any]] = {}
    omitted_by_slice: dict[str, list[OmitRegionSelection]] = {slice_info.key: [] for slice_info in dataset.slices}

    for slice_info in dataset.slices:
        omit_mask = omit_masks_by_slice.get(slice_info.key)
        if omit_mask is None or not np.any(omit_mask):
            continue
        context = _slice_context(dataset, slice_info, context_cache)
        selected = {
            (int(selection.display_code), int(selection.component_label))
            for selection in _selections_from_mask(context, omit_mask)
        }
        omitted_by_slice[slice_info.key] = [
            OmitRegionSelection(display_code=display_code, component_label=component_label)
            for display_code, component_label in sorted(selected)
        ]

    session.omitted_regions_by_slice = omitted_by_slice
    return session


def _stroke_mask(
    *,
    width: int,
    height: int,
    stroke: Any,
    source_size: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Rasterize one QDFevo_2_AtlasFitter omit stroke into an image-space boolean mask."""

    if not isinstance(stroke, dict):
        return None
    raw_points = stroke.get("points", [])
    if not isinstance(raw_points, list) or not raw_points:
        return None
    points: list[tuple[float, float]] = []
    for item in raw_points:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            x = float(item[0])
            y = float(item[1])
        except (TypeError, ValueError):
            continue
        points.append((x, y))
    if not points:
        return None

    try:
        brush_size = max(1, int(round(float(stroke.get("size", 1.0)))))
    except (TypeError, ValueError):
        brush_size = 1

    if source_size is not None:
        source_width, source_height = source_size
        if source_width > 0 and source_height > 0:
            scale_x = float(width) / float(source_width)
            scale_y = float(height) / float(source_height)
            if not math.isclose(scale_x, 1.0) or not math.isclose(scale_y, 1.0):
                points = [(x * scale_x, y * scale_y) for x, y in points]
                brush_size = max(1, int(round(float(brush_size) * ((scale_x + scale_y) * 0.5))))
    radius = max(1, int(round(brush_size / 2.0)))

    canvas = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(canvas)
    mode = str(stroke.get("mode", "brush")).strip().lower()
    if mode == "polygon" and len(points) >= 3:
        draw.polygon(points, fill=255)
        # Keep a visible/selected border consistent with the fitter display.
        draw.line([*points, points[0]], fill=255, width=max(1, brush_size))
    elif len(points) == 1:
        x0, y0 = points[0]
        draw.ellipse((x0 - radius, y0 - radius, x0 + radius, y0 + radius), fill=255)
    else:
        draw.line(points, fill=255, width=brush_size)
        for x0, y0 in points:
            draw.ellipse((x0 - radius, y0 - radius, x0 + radius, y0 + radius), fill=255)
    return np.asarray(canvas, dtype=np.uint8) > 0


def _selections_from_mask(context: dict[str, Any], omit_mask: np.ndarray) -> list[OmitRegionSelection]:
    """Return connected atlas patches touched by one omit mask."""

    atlas_plane = np.asarray(context["atlas_plane"], dtype=np.int64)
    if omit_mask.shape != atlas_plane.shape:
        return []
    touched_codes = np.unique(atlas_plane[omit_mask])
    selections: list[OmitRegionSelection] = []
    for display_code in touched_codes.tolist():
        code = int(display_code)
        if code == 0:
            continue
        component_map = _component_map_for_display_code(context, code)
        labels = np.unique(component_map[omit_mask & (atlas_plane == code)])
        for label in labels.tolist():
            component_label = int(label)
            if component_label <= 0:
                continue
            selections.append(OmitRegionSelection(display_code=code, component_label=component_label))
    return selections


def _write_omit_plane_to_overlay(
    dataset: OverlayDataset,
    omit_masks_by_slice: dict[str, np.ndarray],
) -> dict[str, object]:
    """Append or refresh an omit-region plane in each overlay TIFF stack and workbook."""

    existing_info = next((info for info in dataset.channels if info.content == OMIT_LAYER_CONTENT), None)
    omit_plane_index = int(existing_info.plane_index) if existing_info is not None else None

    for slice_info in dataset.slices:
        stack = tifffile.imread(slice_info.overlay_path)
        if stack.ndim == 2:
            stack = stack[None, ...]
        if stack.ndim != 3:
            raise ValueError(f"Expected 3D overlay stack in {slice_info.overlay_path}, got shape {stack.shape}")
        omit_mask = np.asarray(
            omit_masks_by_slice.get(slice_info.key, np.zeros(stack.shape[1:], dtype=bool)),
            dtype=bool,
        ).astype(np.int32, copy=False) * int(OMIT_PLANE_VALUE)
        if omit_plane_index is None:
            updated_stack = np.concatenate([stack, omit_mask[None, ...]], axis=0)
        else:
            updated_stack = np.asarray(stack).copy()
            if omit_plane_index >= updated_stack.shape[0]:
                pad_count = omit_plane_index - updated_stack.shape[0] + 1
                updated_stack = np.concatenate(
                    [updated_stack, np.zeros((pad_count, *updated_stack.shape[1:]), dtype=updated_stack.dtype)],
                    axis=0,
                )
            updated_stack[omit_plane_index] = omit_mask.astype(updated_stack.dtype, copy=False)
        tifffile.imwrite(slice_info.overlay_path, updated_stack, compression="adobe_deflate")

    updated_channel_maps = _ensure_omit_entry_in_all_channel_maps(dataset)
    return {
        "overlay_omit_layer_content": OMIT_LAYER_CONTENT,
        "overlay_omit_plane_count": int(sum(1 for mask in omit_masks_by_slice.values() if np.any(mask))),
        "overlay_omit_channel_maps_updated": updated_channel_maps,
    }


def _ensure_omit_entry_in_all_channel_maps(dataset: OverlayDataset) -> list[str]:
    """Add omit-region metadata to every channel-map workbook in the overlay folder."""

    candidate_paths = [
        *sorted(dataset.overlay_dir.glob("*multichannel_channel_maps.xlsx")),
        *sorted(dataset.overlay_dir.glob("*multichannel_channel_maps.csv")),
    ]
    if dataset.channel_map_path not in candidate_paths:
        candidate_paths.insert(0, dataset.channel_map_path)

    updated_paths: list[str] = []
    seen: set[Path] = set()
    for channel_map_path in candidate_paths:
        resolved = channel_map_path.resolve()
        if resolved in seen or not channel_map_path.exists():
            continue
        seen.add(resolved)
        channel_map_frame = _read_channel_map_frame(channel_map_path)
        if OMIT_LAYER_CONTENT in channel_map_frame.get("content", pd.Series(dtype=str)).astype(str).tolist():
            continue
        next_index = int(pd.to_numeric(channel_map_frame["channel_index"], errors="coerce").fillna(0).max()) + 1
        channel_map_frame = pd.concat(
            [
                channel_map_frame,
                pd.DataFrame(
                    [
                        {
                            "channel_index": next_index,
                            "channel_name": f"CH{next_index}",
                            "content": OMIT_LAYER_CONTENT,
                            "source_channel": "",
                            "notes": "Omit-region mask imported from QDFevo_2_AtlasFitter omit state.",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        channel_map_frame = channel_map_frame.sort_values("channel_index").reset_index(drop=True)
        _write_channel_map_frame(channel_map_path, channel_map_frame)
        updated_paths.append(str(channel_map_path))
    return updated_paths


def _read_channel_map_frame(channel_map_path: Path) -> pd.DataFrame:
    if channel_map_path.suffix.lower() == ".csv":
        return pd.read_csv(channel_map_path)
    return pd.read_excel(channel_map_path, sheet_name="overlay_stack")


def _write_channel_map_frame(channel_map_path: Path, channel_map_frame: pd.DataFrame) -> None:
    if channel_map_path.suffix.lower() == ".csv":
        channel_map_frame.to_csv(channel_map_path, index=False)
    else:
        with pd.ExcelWriter(channel_map_path, engine="openpyxl") as writer:
            channel_map_frame.to_excel(writer, sheet_name="overlay_stack", index=False)


def _overwrite_tables_with_omit(
    output_dir: Path,
    dataset: OverlayDataset,
    omit_masks_by_slice: dict[str, np.ndarray],
) -> dict[str, object]:
    """Overwrite main QDF2 tables using mask-based omit logic."""

    cell_path = output_dir / "cell_level.csv"
    region_path = output_dir / "region_summary.csv"
    section_path = output_dir / "section_summary.csv"
    if not cell_path.exists() or not region_path.exists() or not section_path.exists():
        return {"omit_tables_updated": False, "reason": "tables_missing"}

    cell_df = pd.read_csv(cell_path)
    region_df = pd.read_csv(region_path)
    section_df = pd.read_csv(section_path)

    flagged_cell = _apply_omit_mask_to_cell_level(cell_df, dataset, omit_masks_by_slice)
    active_cell = flagged_cell.loc[flagged_cell["omit_flag"].eq(0)].copy()
    exact_overlap = (
        "overlap_group_id" in active_cell.columns
        and active_cell["overlap_group_id"].fillna("").astype(str).str.strip().ne("").any()
    )
    affected_group_ids = set(
        flagged_cell.loc[flagged_cell["omit_flag"].eq(1), "overlap_group_id"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .tolist()
    )
    if exact_overlap and affected_group_ids:
        active_cell = _recompute_overlap_membership(active_cell, affected_group_ids=affected_group_ids)

    section_omit = _build_section_summary_with_omit(active_cell, section_df)
    region_omit = _build_region_summary_with_omit_masks(
        active_cell,
        region_df,
        section_omit,
        dataset,
        omit_masks_by_slice,
        exact_overlap=bool(exact_overlap),
    )

    flagged_cell.to_csv(cell_path, index=False)
    region_omit.to_csv(region_path, index=False)
    section_omit.to_csv(section_path, index=False)
    return {
        "omit_tables_updated": True,
        "cell_level_rows_after_omit": int(len(flagged_cell)),
        "omitted_cell_count": int(flagged_cell["omit_flag"].sum()),
        "region_summary_rows_after_omit": int(len(region_omit)),
        "section_summary_rows_after_omit": int(len(section_omit)),
    }


def _roi_plane_indices_by_channel(dataset: OverlayDataset) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for info in dataset.channels:
        if info.is_cell_roi and info.source_channel:
            lookup[str(info.source_channel).upper()] = int(info.plane_index)
    return lookup


def _sample_component_labels(label_map: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    labels = np.zeros(len(x), dtype=np.int32)
    valid_xy = (x >= 0) & (x < label_map.shape[1]) & (y >= 0) & (y < label_map.shape[0])
    if np.any(valid_xy):
        labels[valid_xy] = label_map[y[valid_xy], x[valid_xy]].astype(np.int32, copy=False)
    unresolved = valid_xy & (labels == 0)
    if not np.any(unresolved):
        return labels
    for radius in (1, 2, 3):
        unresolved_idx = np.flatnonzero(unresolved)
        if unresolved_idx.size == 0:
            break
        for idx in unresolved_idx.tolist():
            x0 = int(x[idx])
            y0 = int(y[idx])
            x_min = max(0, x0 - radius)
            x_max = min(label_map.shape[1], x0 + radius + 1)
            y_min = max(0, y0 - radius)
            y_max = min(label_map.shape[0], y0 + radius + 1)
            window = label_map[y_min:y_max, x_min:x_max]
            positive = window[window > 0]
            if positive.size == 0:
                continue
            values, counts = np.unique(positive, return_counts=True)
            labels[idx] = int(values[np.argmax(counts)])
            unresolved[idx] = False
    return labels


def _apply_omit_mask_to_cell_level(
    cell_df: pd.DataFrame,
    dataset: OverlayDataset,
    omit_masks_by_slice: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Flag cells whose ROI overlaps the omit mask, preserving all cell rows."""

    working = cell_df.copy()
    working["animal_id"] = working["animal_id"].astype(str)
    working["section_id"] = working["section_id"].astype(str)
    working["image_channel"] = working["image_channel"].astype(str)
    working["centroid_x_px"] = pd.to_numeric(working["centroid_x_px"], errors="coerce")
    working["centroid_y_px"] = pd.to_numeric(working["centroid_y_px"], errors="coerce")
    working["omit_flag"] = 0
    working["omit_source"] = ""

    roi_plane_indices = _roi_plane_indices_by_channel(dataset)
    slice_lookup = {(slice_info.animal_id, slice_info.section_id): slice_info for slice_info in dataset.slices}

    for (animal_id, section_id), index_labels in working.groupby(["animal_id", "section_id"], sort=False).groups.items():
        slice_info = slice_lookup.get((str(animal_id), str(section_id)))
        if slice_info is None:
            continue
        omit_mask = np.asarray(omit_masks_by_slice.get(slice_info.key), dtype=bool) if slice_info.key in omit_masks_by_slice else None
        if omit_mask is None or not np.any(omit_mask):
            continue

        section_frame = working.loc[pd.Index(index_labels)]

        for image_channel, channel_frame in section_frame.groupby("image_channel", sort=False):
            x = np.rint(channel_frame["centroid_x_px"].fillna(-1).to_numpy(dtype=np.float64)).astype(np.int32)
            y = np.rint(channel_frame["centroid_y_px"].fillna(-1).to_numpy(dtype=np.float64)).astype(np.int32)
            valid_xy = (x >= 0) & (x < omit_mask.shape[1]) & (y >= 0) & (y < omit_mask.shape[0])
            omit_flags = np.zeros(len(channel_frame), dtype=bool)
            if np.any(valid_xy):
                omit_flags[valid_xy] = omit_mask[y[valid_xy], x[valid_xy]]

            plane_index = roi_plane_indices.get(str(image_channel).upper())
            if plane_index is not None:
                try:
                    roi_plane = np.asarray(_read_overlay_plane(slice_info.overlay_path, int(plane_index)) > 0, dtype=np.uint8)
                except Exception:
                    LOG.exception("Failed reading ROI plane %s from %s", plane_index, slice_info.overlay_path)
                    roi_plane = None
            else:
                roi_plane = None
            if roi_plane is not None:
                labels, _ = ndimage.label(roi_plane, structure=np.ones((3, 3), dtype=np.uint8))
                component_labels = _sample_component_labels(labels, x, y)
                touched_components = np.unique(labels[omit_mask])
                touched_components = touched_components[touched_components > 0]
                if touched_components.size:
                    component_touched = np.zeros(int(touched_components.max()) + 1, dtype=bool)
                    component_touched[touched_components] = True
                    valid_components = (component_labels > 0) & (component_labels < len(component_touched))
                    omit_flags[valid_components] = component_touched[component_labels[valid_components]]

            omit_indices = channel_frame.index[omit_flags]
            if len(omit_indices) == 0:
                continue
            working.loc[omit_indices, "omit_flag"] = 1
            working.loc[omit_indices, "omit_source"] = "qdf1_atlasfitter_omit_mask"

    return working


def _build_section_summary_with_omit(
    active_cell_df: pd.DataFrame,
    base_section: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild section_summary from non-omitted cells while preserving schema."""

    rows: list[dict[str, object]] = []
    for record in base_section.to_dict(orient="records"):
        animal_id = str(record.get("animal_id", ""))
        section_id = str(record.get("section_id", ""))
        image_channel = str(record.get("image_channel", ""))
        row = dict(record)
        mask = (
            active_cell_df["animal_id"].astype(str).eq(animal_id)
            & active_cell_df["section_id"].astype(str).eq(section_id)
            & active_cell_df["image_channel"].astype(str).eq(image_channel)
        )
        group = active_cell_df.loc[mask]
        region_ids = pd.to_numeric(group.get("region_id", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int)
        hemispheres = group.get("hemisphere", pd.Series(dtype=str)).fillna("").astype(str).str.lower()
        row["n_detected_cells"] = int(len(group))
        row["n_unassigned_cells"] = int(region_ids.eq(0).sum())
        row["left_count"] = int(hemispheres.eq("left").sum())
        row["right_count"] = int(hemispheres.eq("right").sum())
        rows.append(row)
    output = pd.DataFrame(rows)
    return output.reindex(columns=list(base_section.columns))


def _remaining_region_area_lookup_from_masks(
    dataset: OverlayDataset,
    omit_masks_by_slice: dict[str, np.ndarray],
    pixel_area_lookup: dict[tuple[str, str], float],
) -> dict[tuple[str, str], dict[tuple[int, str], float]]:
    """Return remaining atlas area after subtracting omit mask pixels only."""

    cache: dict[str, dict[str, Any]] = {}
    area_lookup_by_section: dict[tuple[str, str], dict[tuple[int, str], float]] = {}
    for slice_info in dataset.slices:
        pixel_area_um2 = float(pixel_area_lookup.get((slice_info.animal_id, slice_info.section_id), 0.0))
        context = _slice_context(dataset, slice_info, cache)
        atlas_plane = np.asarray(context["atlas_plane"], dtype=np.int64)
        omit_mask = np.asarray(omit_masks_by_slice.get(slice_info.key, np.zeros(atlas_plane.shape, dtype=bool)), dtype=bool)
        remaining_codes = atlas_plane[(atlas_plane != 0) & ~omit_mask]
        lookup: dict[tuple[int, str], float] = {}
        if remaining_codes.size:
            display_codes, total_counts = np.unique(remaining_codes, return_counts=True)
            for display_code, count in zip(display_codes, total_counts, strict=False):
                code = int(display_code)
                region_info = dataset.region_lookup.get(code)
                region_id = int(region_info.region_id) if region_info is not None else abs(code)
                total_key = (int(region_id), "total")
                lookup[total_key] = lookup.get(total_key, 0.0) + (float(count) * pixel_area_um2)
                hemisphere_name = str(region_info.hemisphere).lower() if region_info is not None else (
                    "left" if code < 0 else "right"
                )
                if hemisphere_name in {"left", "right"}:
                    key = (int(region_id), hemisphere_name)
                    lookup[key] = lookup.get(key, 0.0) + (float(count) * pixel_area_um2)
        area_lookup_by_section[(slice_info.animal_id, slice_info.section_id)] = lookup
    return area_lookup_by_section


def _build_region_summary_with_omit_masks(
    active_cell_df: pd.DataFrame,
    base_region: pd.DataFrame,
    section_summary: pd.DataFrame,
    dataset: OverlayDataset,
    omit_masks_by_slice: dict[str, np.ndarray],
    *,
    exact_overlap: bool,
) -> pd.DataFrame:
    """Rebuild region_summary from non-omitted cells and mask-reduced region area."""

    base_columns = list(base_region.columns)
    overlap_labels = sorted({column[:-8] for column in base_columns if column.endswith("_n_cells")})
    overlap_defaults: dict[str, object] = {}
    for label in overlap_labels:
        overlap_defaults[f"{label}_n_cells"] = 0
        overlap_defaults[f"{label}_mean_integrated_intensity"] = None
        overlap_defaults[f"{label}_mean_cell_area"] = None

    pixel_area_lookup = (
        section_summary.groupby(["animal_id", "section_id"], as_index=False)["pixel_area_um2"].first()
        .set_index(["animal_id", "section_id"])["pixel_area_um2"]
        .to_dict()
    )
    area_lookup_by_section = _remaining_region_area_lookup_from_masks(dataset, omit_masks_by_slice, pixel_area_lookup)
    empty_frame = active_cell_df.iloc[0:0].copy()
    section_groups = {
        (str(animal_id), str(section_id)): frame.copy()
        for (animal_id, section_id), frame in active_cell_df.groupby(["animal_id", "section_id"], sort=False)
    }
    overlap_metrics_cache: dict[tuple[str, str], dict[tuple[int, str], dict[str, object]]] = {}

    rows: list[dict[str, object]] = []
    for record in base_region.to_dict(orient="records"):
        animal_id = str(record.get("animal_id", ""))
        section_id = str(record.get("section_id", ""))
        image_channel = str(record.get("image_channel", ""))
        region_id = int(pd.to_numeric(pd.Series([record.get("region_id", 0)]), errors="coerce").fillna(0).iloc[0])
        hemisphere = str(record.get("hemisphere", ""))

        section_frame = section_groups.get((animal_id, section_id), empty_frame)
        overlap_metrics = overlap_metrics_cache.setdefault(
            (animal_id, section_id),
            _compute_overlap_metrics(section_frame, overlap_labels, exact_overlap=exact_overlap),
        )
        channel_frame = section_frame.loc[section_frame["image_channel"].astype(str).eq(image_channel)]
        region_ids = pd.to_numeric(channel_frame.get("region_id", pd.Series(dtype=int)), errors="coerce").fillna(0).astype(int)
        hemispheres = channel_frame.get("hemisphere", pd.Series(dtype=str)).fillna("").astype(str).str.lower()
        if hemisphere == "total":
            group = channel_frame.loc[region_ids.eq(region_id)]
        else:
            group = channel_frame.loc[region_ids.eq(region_id) & hemispheres.eq(str(hemisphere).lower())]

        row = dict(record)
        row["n_cells"] = int(len(group))
        row["total_integrated_intensity"] = float(group["integrated_intensity"].sum()) if not group.empty else 0.0
        row["mean_integrated_intensity"] = float(group["mean_intensity"].mean()) if not group.empty else None
        row["mean_cell_area"] = float(group["area_px"].mean()) if not group.empty else None
        area_um2 = float(area_lookup_by_section.get((animal_id, section_id), {}).get((region_id, hemisphere), 0.0))
        row["region_area_um2"] = area_um2
        row["density_if_possible"] = float(len(group) / (area_um2 / 1_000_000.0)) if area_um2 > 0 and len(group) > 0 else None
        patch_summary = _region_patch_summary_from_frame(group)
        row["atlas_patch_count"] = patch_summary["atlas_patch_count"]
        row["atlas_display_codes"] = patch_summary["atlas_display_codes"]
        row["atlas_patch_components"] = patch_summary["atlas_patch_components"]
        row["atlas_patch_ids"] = patch_summary["atlas_patch_ids"]
        for key, value in overlap_defaults.items():
            row[key] = value
        for key, value in overlap_metrics.get((region_id, hemisphere), {}).items():
            row[key] = value
        rows.append(row)

    output = pd.DataFrame(rows)
    if output.empty:
        return pd.DataFrame(columns=base_columns)
    for column in base_columns:
        if column not in output.columns:
            output[column] = pd.NA
    return output.reindex(columns=base_columns)
