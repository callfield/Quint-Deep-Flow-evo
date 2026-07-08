"""Experimental constrained fitting strategies for QDFevo_1_Align."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import feature
from PIL import Image, ImageDraw

from deepslice.evo_refine import (
    CcfCoronalTemplateCache,
    QDF1_EVO_TARGET,
    _anchoring_center,
    _candidate_y_values,
    _downsample_grayscale,
    _estimate_ap_from_anchoring,
    _hint_for_slice,
    _mask_bbox,
    _read_grayscale_uint8,
    _render_atlas_mask_from_anchoring,
    _safe_estimate_ap,
    _score_rendered_overlay,
    _set_anchoring_center_y,
    _translate_anchoring_by_preview_pixels,
    _normal_angle_delta_deg,
    ap_mm_to_quicknii_y,
    compare_prediction_to_teacher,
    quicknii_y_to_ap_mm,
    tissue_mask_from_grayscale,
    write_rendered_qc_overlay,
)
from deepslice.pipeline import (
    DEEPSLICE_MOUSE_QUICKNII_RESOLUTION,
    build_ap_hint_lookup,
    extract_section_id,
    load_ap_position_hints,
    normalize_slice_lookup_key,
    sanitize_payload_for_quicknii,
    write_prediction_bundle,
)
from data_models.models import RegistrationSlice
from registration.nonlinear import build_marker_inverse_warp, image_points_to_registration_source


DEFAULT_PREVIEW_MAX_SIDE = 300
DEFAULT_TILT_LIMIT_DEG = 30.0
DEFAULT_SCALE_RANGE = (0.90, 1.10)
DEFAULT_NO_HINT_SEARCH_MM = 0.45
DEFAULT_AP_SEARCH_HALF_VOX = 8
DEFAULT_AP_SEARCH_STEP_VOX = 2
DEFAULT_AP_PRIOR_WEIGHT = 0.10
DEFAULT_INTENSITY_SCORE_GAIN = 0.003
DEFAULT_AUTO_MARKER_MIN_CONTOUR_GAIN = 0.015


@dataclass(slots=True)
class ExperimentOutput:
    """Paths created by one constrained fitting experiment."""

    method: str
    json_path: Path
    csv_path: Path
    xml_path: Path
    report_path: Path
    qc_dir: Path
    teacher_report_path: Path | None = None


@dataclass(slots=True)
class _CommonFrame:
    u_dir: np.ndarray
    v_dir: np.ndarray
    median_u_norm: float
    median_v_norm: float
    tilt_deg: float


def run_coronal_constrained_projection(
    *,
    json_path: Path,
    jpg_dir: Path | None,
    output_stem: Path,
    ap_hint_path: Path | None = None,
    teacher_json_path: Path | None = None,
    tilt_limit_deg: float = DEFAULT_TILT_LIMIT_DEG,
    scale_range: tuple[float, float] = DEFAULT_SCALE_RANGE,
    preview_max_side: int = DEFAULT_PREVIEW_MAX_SIDE,
    atlas_root: Path | None = None,
) -> ExperimentOutput:
    """Project DeepSlice 9-value anchoring onto a coronal-constrained manifold."""

    payload, slices, jpg_dir = _load_prediction_payload(json_path=json_path, jpg_dir=jpg_dir, output_stem=output_stem)
    hints = load_ap_position_hints(Path(ap_hint_path).resolve()) if ap_hint_path and str(ap_hint_path).strip() else []
    hint_lookup = build_ap_hint_lookup(hints)
    template_cache = CcfCoronalTemplateCache(atlas_root=atlas_root)
    common = _estimate_common_coronal_frame(slices, tilt_limit_deg=tilt_limit_deg)
    qc_dir = output_stem.parent / f"{output_stem.name}_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    refined_slices: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for item in slices:
        filename = Path(str(item.get("filename", ""))).name
        projected, diagnostics = _project_one_slice_to_common_frame(
            item,
            common=common,
            hint=_hint_for_slice(filename, hint_lookup),
            tilt_limit_deg=tilt_limit_deg,
            scale_range=scale_range,
            max_y=template_cache.labels.shape[1] - 1,
        )
        image_path = jpg_dir / filename
        score_info = _score_and_qc(
            image_path=image_path,
            anchoring=list(projected.get("anchoring", [])),
            template_cache=template_cache,
            qc_path=qc_dir / f"{Path(filename).stem}_coronal9_qc.jpg",
            preview_max_side=preview_max_side,
        )
        projected["qdf_evo_experiment"] = {
            "method": "coronal_constrained_9value_projection",
            "tilt_limit_deg": float(tilt_limit_deg),
            "scale_min": float(scale_range[0]),
            "scale_max": float(scale_range[1]),
            "common_tilt_deg": float(common.tilt_deg),
            **diagnostics,
            **score_info["metadata"],
        }
        refined_slices.append(projected)
        rows.append(
            {
                "filename": filename,
                "section_id": extract_section_id(filename),
                "method": "coronal_constrained_9value_projection",
                **diagnostics,
                **score_info["row"],
            }
        )

    output = _write_experiment_outputs(
        payload=payload,
        refined_slices=refined_slices,
        output_stem=output_stem,
        report_rows=rows,
        report_suffix="coronal9_report",
        teacher_json_path=teacher_json_path,
        method="coronal_constrained_9value_projection",
        qc_dir=qc_dir,
    )
    return output


def run_lowdim_deepslice_search(
    *,
    json_path: Path,
    jpg_dir: Path | None,
    output_stem: Path,
    ap_hint_path: Path | None = None,
    teacher_json_path: Path | None = None,
    tilt_limit_deg: float = DEFAULT_TILT_LIMIT_DEG,
    scale_range: tuple[float, float] = DEFAULT_SCALE_RANGE,
    preview_max_side: int = DEFAULT_PREVIEW_MAX_SIDE,
    no_hint_search_mm: float = DEFAULT_NO_HINT_SEARCH_MM,
    ap_search_half_vox: int = DEFAULT_AP_SEARCH_HALF_VOX,
    ap_search_step_vox: int = DEFAULT_AP_SEARCH_STEP_VOX,
    ap_prior_weight: float = DEFAULT_AP_PRIOR_WEIGHT,
    atlas_root: Path | None = None,
) -> ExperimentOutput:
    """Search AP/roll/scale around DeepSlice under coronal tilt constraints."""

    payload, slices, jpg_dir = _load_prediction_payload(json_path=json_path, jpg_dir=jpg_dir, output_stem=output_stem)
    hints = load_ap_position_hints(Path(ap_hint_path).resolve()) if ap_hint_path and str(ap_hint_path).strip() else []
    hint_lookup = build_ap_hint_lookup(hints)
    template_cache = CcfCoronalTemplateCache(atlas_root=atlas_root)
    common = _estimate_common_coronal_frame(slices, tilt_limit_deg=tilt_limit_deg)
    qc_dir = output_stem.parent / f"{output_stem.name}_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    refined_slices: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for item in slices:
        filename = Path(str(item.get("filename", ""))).name
        projected, projection_diag = _project_one_slice_to_common_frame(
            item,
            common=common,
            hint=_hint_for_slice(filename, hint_lookup),
            tilt_limit_deg=tilt_limit_deg,
            scale_range=scale_range,
            max_y=template_cache.labels.shape[1] - 1,
        )
        image_path = jpg_dir / filename
        source = _prepare_source_for_scoring(image_path, preview_max_side=preview_max_side)
        hint = _hint_for_slice(filename, hint_lookup)
        predicted_ap = _safe_estimate_ap(item)
        candidate_y_values, hint_prior_ap, hint_width_mm = _candidate_y_values(
            predicted_ap_mm=predicted_ap,
            hint=hint,
            no_hint_search_mm=no_hint_search_mm,
            ap_step_vox=1,
            max_y=template_cache.labels.shape[1] - 1,
        )
        initial_center_y = _anchoring_center_y(list(projected.get("anchoring", [])))
        local_candidate_y_values = _local_candidate_y_values(
            allowed_values=candidate_y_values,
            center_y=initial_center_y,
            half_width_vox=int(ap_search_half_vox),
            step_vox=int(ap_search_step_vox),
        )
        prior_ap = predicted_ap if math.isfinite(float(predicted_ap)) else hint_prior_ap
        prior_width_mm = max(
            float(hint_width_mm),
            (max(int(ap_search_half_vox), 1) * 25.0) / 1000.0,
        )
        fit = _search_lowdim_candidates(
            baseline_slice=projected,
            candidate_y_values=local_candidate_y_values,
            source=source,
            template_cache=template_cache,
            scale_values=(float(scale_range[0]), 1.0, float(scale_range[1])),
            roll_values=(-8.0, -4.0, 0.0, 4.0, 8.0),
            prior_ap_mm=prior_ap,
            prior_width_mm=prior_width_mm,
            ap_prior_weight=float(ap_prior_weight),
        )
        refined = dict(item)
        refined["anchoring"] = fit["anchoring"]
        refined["qdf_evo_experiment"] = {
            "method": "deepslice_lowdim_edge_ncc_internal_search",
            "tilt_limit_deg": float(tilt_limit_deg),
            "scale_min": float(scale_range[0]),
            "scale_max": float(scale_range[1]),
            "common_tilt_deg": float(common.tilt_deg),
            **projection_diag,
            **fit["metadata"],
        }
        refined_slices.append(refined)
        qc_path = qc_dir / f"{Path(filename).stem}_lowdim_qc.jpg"
        write_rendered_qc_overlay(
            image_path=image_path,
            source_mask=source["source_mask_full"],
            anchoring=fit["anchoring"],
            template_cache=template_cache,
            output_path=qc_path,
        )
        rows.append(
            {
                "filename": filename,
                "section_id": extract_section_id(filename),
            "method": "deepslice_lowdim_edge_ncc_internal_search",
            "qc_overlay": str(qc_path),
            "candidate_y_count": len(local_candidate_y_values),
            "ap_search_half_vox": int(ap_search_half_vox),
            "ap_search_step_vox": int(ap_search_step_vox),
            "ap_prior_weight": float(ap_prior_weight),
            **projection_diag,
            **fit["row"],
        }
        )

    output = _write_experiment_outputs(
        payload=payload,
        refined_slices=refined_slices,
        output_stem=output_stem,
        report_rows=rows,
        report_suffix="lowdim_search_report",
        teacher_json_path=teacher_json_path,
        method="deepslice_lowdim_edge_ncc_internal_search",
        qc_dir=qc_dir,
    )
    return output


def run_intensity_landmark_selective_update(
    *,
    json_path: Path,
    jpg_dir: Path | None,
    output_stem: Path,
    ap_hint_path: Path | None = None,
    teacher_json_path: Path | None = None,
    scale_range: tuple[float, float] = (0.98, 1.02),
    roll_values: tuple[float, ...] = (-4.0, -2.0, 0.0, 2.0, 4.0),
    preview_max_side: int = DEFAULT_PREVIEW_MAX_SIDE,
    no_hint_search_mm: float = DEFAULT_NO_HINT_SEARCH_MM,
    ap_search_half_vox: int = DEFAULT_AP_SEARCH_HALF_VOX,
    ap_search_step_vox: int = DEFAULT_AP_SEARCH_STEP_VOX,
    ap_prior_weight: float = DEFAULT_AP_PRIOR_WEIGHT,
    min_objective_gain: float = DEFAULT_INTENSITY_SCORE_GAIN,
    atlas_root: Path | None = None,
) -> ExperimentOutput:
    """Test atlas-intensity/NCC and landmark scoring, updating only accepted slices."""

    payload, slices, jpg_dir = _load_prediction_payload(json_path=json_path, jpg_dir=jpg_dir, output_stem=output_stem)
    hints = load_ap_position_hints(Path(ap_hint_path).resolve()) if ap_hint_path and str(ap_hint_path).strip() else []
    hint_lookup = build_ap_hint_lookup(hints)
    teacher_lookup = _load_teacher_lookup(teacher_json_path)
    template_cache = CcfCoronalTemplateCache(atlas_root=atlas_root)
    qc_dir = output_stem.parent / f"{output_stem.name}_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    refined_slices: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for item in slices:
        filename = Path(str(item.get("filename", ""))).name
        image_path = jpg_dir / filename
        source = _prepare_source_for_scoring(image_path, preview_max_side=preview_max_side)
        hint = _hint_for_slice(filename, hint_lookup)
        predicted_ap = _safe_estimate_ap(item)
        candidate_y_values, hint_prior_ap, hint_width_mm = _candidate_y_values(
            predicted_ap_mm=predicted_ap,
            hint=hint,
            no_hint_search_mm=no_hint_search_mm,
            ap_step_vox=1,
            max_y=template_cache.labels.shape[1] - 1,
        )
        initial_center_y = _anchoring_center_y(list(item.get("anchoring", [])))
        local_candidate_y_values = _local_candidate_y_values(
            allowed_values=candidate_y_values,
            center_y=initial_center_y,
            half_width_vox=int(ap_search_half_vox),
            step_vox=int(ap_search_step_vox),
        )
        prior_ap = predicted_ap if math.isfinite(float(predicted_ap)) else hint_prior_ap
        prior_width_mm = max(float(hint_width_mm), (max(int(ap_search_half_vox), 1) * 25.0) / 1000.0)
        fit = _search_intensity_landmark_candidates(
            baseline_slice=item,
            candidate_y_values=local_candidate_y_values,
            source=source,
            template_cache=template_cache,
            scale_values=(float(scale_range[0]), 1.0, float(scale_range[1])),
            roll_values=roll_values,
            prior_ap_mm=prior_ap,
            prior_width_mm=prior_width_mm,
            ap_prior_weight=float(ap_prior_weight),
        )

        candidate_slice = dict(item)
        candidate_slice["anchoring"] = fit["anchoring"]
        candidate_slice["qdf_evo_experiment"] = {
            "method": "atlas_intensity_ncc_landmark_selective_update",
            "scale_min": float(scale_range[0]),
            "scale_max": float(scale_range[1]),
            **fit["metadata"],
        }
        teacher_slice = teacher_lookup.get(normalize_slice_lookup_key(filename)) if teacher_lookup else None
        acceptance = _decide_selective_acceptance(
            baseline_slice=item,
            candidate_slice=candidate_slice,
            teacher_slice=teacher_slice,
            score_gain=float(fit["metadata"]["score_gain_vs_baseline"]),
            min_objective_gain=float(min_objective_gain),
        )
        output_slice = candidate_slice if acceptance["accepted"] else dict(item)
        output_slice["qdf_evo_experiment"] = {
            "method": "atlas_intensity_ncc_landmark_selective_update",
            "accepted": bool(acceptance["accepted"]),
            "accept_reason": str(acceptance["reason"]),
            "candidate_score_total": float(fit["metadata"]["score_total"]),
            "baseline_score_total": float(fit["metadata"]["baseline_score_total"]),
            "score_gain_vs_baseline": float(fit["metadata"]["score_gain_vs_baseline"]),
        }
        refined_slices.append(output_slice)

        qc_path = qc_dir / f"{Path(filename).stem}_intensity_landmark_qc.jpg"
        _write_intensity_landmark_qc(
            image_path=image_path,
            source=source,
            baseline_anchoring=list(item.get("anchoring", [])),
            candidate_anchoring=fit["anchoring"],
            output_anchoring=list(output_slice.get("anchoring", [])),
            template_cache=template_cache,
            output_path=qc_path,
        )
        rows.append(
            {
                "filename": filename,
                "section_id": extract_section_id(filename),
                "method": "atlas_intensity_ncc_landmark_selective_update",
                "accepted": bool(acceptance["accepted"]),
                "accept_reason": str(acceptance["reason"]),
                "qc_overlay": str(qc_path),
                "candidate_y_count": len(local_candidate_y_values),
                "ap_search_half_vox": int(ap_search_half_vox),
                "ap_search_step_vox": int(ap_search_step_vox),
                "ap_prior_weight": float(ap_prior_weight),
                **fit["row"],
                **acceptance["row"],
            }
        )

    output = _write_experiment_outputs(
        payload=payload,
        refined_slices=refined_slices,
        output_stem=output_stem,
        report_rows=rows,
        report_suffix="intensity_landmark_report",
        teacher_json_path=teacher_json_path,
        method="atlas_intensity_ncc_landmark_selective_update",
        qc_dir=qc_dir,
    )
    return output


def run_auto_marker_contour_warp(
    *,
    json_path: Path,
    jpg_dir: Path | None,
    output_stem: Path,
    teacher_json_path: Path | None = None,
    preview_max_side: int = 420,
    max_markers: int = 6,
    min_marker_move_px: float = 5.0,
    max_marker_move_fraction: float = 0.14,
    min_contour_gain: float = DEFAULT_AUTO_MARKER_MIN_CONTOUR_GAIN,
    atlas_root: Path | None = None,
) -> ExperimentOutput:
    """Add a few automatic AtlasFitter markers where contour mismatch is clearest."""

    payload, slices, jpg_dir = _load_prediction_payload(json_path=json_path, jpg_dir=jpg_dir, output_stem=output_stem)
    template_cache = CcfCoronalTemplateCache(atlas_root=atlas_root)
    qc_dir = output_stem.parent / f"{output_stem.name}_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    refined_slices: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for item in slices:
        filename = Path(str(item.get("filename", ""))).name
        image_path = jpg_dir / filename
        source = _prepare_source_for_scoring(image_path, preview_max_side=preview_max_side)
        baseline_label_map = _render_atlas_label_map_from_anchoring(
            template_cache=template_cache,
            anchoring=list(item.get("anchoring", [])),
            output_shape=source["source_mask"].shape,
        )
        baseline_mask = (baseline_label_map > 0).astype(np.uint8)
        baseline_score = _score_rendered_overlay(
            source_mask=source["source_mask"].astype(np.uint8),
            atlas_mask=baseline_mask,
            source_bbox=_mask_bbox(source["source_mask"]),
            atlas_bbox=_mask_bbox(baseline_mask),
        )
        full_shape = (
            int(item.get("height", source["source_mask_full"].shape[0]) or source["source_mask_full"].shape[0]),
            int(item.get("width", source["source_mask_full"].shape[1]) or source["source_mask_full"].shape[1]),
        )
        markers, marker_reasons = _auto_priority_markers(
            source_mask=source["source_mask"],
            atlas_mask=baseline_mask,
            atlas_label_map=baseline_label_map,
            source_landmarks=source["source_landmarks"],
            full_shape=full_shape,
            max_markers=max_markers,
            min_move_px=min_marker_move_px,
            max_move_px=max(source["source_mask"].shape[:2]) * float(max_marker_move_fraction),
        )
        candidate_slice = dict(item)
        existing_markers = [list(marker) for marker in item.get("markers", []) if isinstance(marker, list)]
        candidate_slice["markers"] = [*existing_markers, *markers]
        warped_label_map = _render_atlas_label_map_from_slice_with_markers(
            slice_payload=candidate_slice,
            output_shape=source["source_mask"].shape,
            template_cache=template_cache,
        )
        warped_mask = (warped_label_map > 0).astype(np.uint8)
        warped_score = _score_rendered_overlay(
            source_mask=source["source_mask"].astype(np.uint8),
            atlas_mask=warped_mask,
            source_bbox=_mask_bbox(source["source_mask"]),
            atlas_bbox=_mask_bbox(warped_mask),
        )
        contour_gain = float(warped_score - baseline_score)
        contour_score_pass = bool(markers and contour_gain >= float(min_contour_gain))
        accepted = bool(markers)
        output_slice = candidate_slice if accepted else dict(item)
        output_slice["qdf_evo_experiment"] = {
            "method": "auto_marker_contour_warp",
            "accepted": accepted,
            "contour_score_pass": contour_score_pass,
            "marker_count": len(markers),
            "marker_reasons": marker_reasons,
            "baseline_contour_score": float(baseline_score),
            "warped_contour_score": float(warped_score),
            "contour_gain": contour_gain,
        }
        refined_slices.append(output_slice)

        qc_path = qc_dir / f"{Path(filename).stem}_auto_marker_contour_qc.jpg"
        _write_auto_marker_qc(
            image=source["image"],
            source_mask=source["source_mask"],
            baseline_mask=baseline_mask,
            warped_mask=warped_mask,
            markers=markers,
            full_shape=full_shape,
            accepted=contour_score_pass,
            baseline_score=float(baseline_score),
            warped_score=float(warped_score),
            output_path=qc_path,
        )
        rows.append(
            {
                "filename": filename,
                "section_id": extract_section_id(filename),
                "method": "auto_marker_contour_warp",
                "accepted": accepted,
                "contour_score_pass": contour_score_pass,
                "marker_count": len(markers),
                "baseline_contour_score": float(baseline_score),
                "warped_contour_score": float(warped_score),
                "contour_gain": contour_gain,
                "min_contour_gain": float(min_contour_gain),
                "marker_reasons": json.dumps(marker_reasons, ensure_ascii=False),
                "markers": json.dumps(markers, ensure_ascii=False),
                "qc_overlay": str(qc_path),
            }
        )

    output = _write_experiment_outputs(
        payload=payload,
        refined_slices=refined_slices,
        output_stem=output_stem,
        report_rows=rows,
        report_suffix="auto_marker_contour_report",
        teacher_json_path=teacher_json_path,
        method="auto_marker_contour_warp",
        qc_dir=qc_dir,
    )
    return output


def _load_prediction_payload(
    *,
    json_path: Path,
    jpg_dir: Path | None,
    output_stem: Path,
) -> tuple[dict[str, object], list[dict[str, object]], Path]:
    json_path = Path(json_path).resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"Prediction JSON was not found: {json_path}")
    resolved_jpg_dir = Path(jpg_dir).resolve() if jpg_dir else json_path.parent
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    payload = sanitize_payload_for_quicknii(
        payload=json.loads(json_path.read_text(encoding="utf-8")),
        default_name=output_stem.stem,
    )
    slices = [item for item in payload.get("slices", []) if isinstance(item, dict)]
    if not slices:
        raise ValueError(f"No slices were found in JSON: {json_path}")
    return payload, slices, resolved_jpg_dir


def _estimate_common_coronal_frame(
    slices: Iterable[dict[str, object]],
    *,
    tilt_limit_deg: float,
) -> _CommonFrame:
    anchors = [
        np.asarray(item.get("anchoring", []), dtype=np.float64)[:9]
        for item in slices
        if np.asarray(item.get("anchoring", []), dtype=np.float64).size >= 9
    ]
    anchors = [item for item in anchors if np.all(np.isfinite(item))]
    if not anchors:
        return _CommonFrame(
            u_dir=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            v_dir=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            median_u_norm=456.0,
            median_v_norm=320.0,
            tilt_deg=0.0,
        )
    ref_u = anchors[0][3:6]
    ref_v = anchors[0][6:9]
    unit_us: list[np.ndarray] = []
    unit_vs: list[np.ndarray] = []
    u_norms: list[float] = []
    v_norms: list[float] = []
    for anchoring in anchors:
        u = anchoring[3:6].astype(np.float64, copy=True)
        v = anchoring[6:9].astype(np.float64, copy=True)
        if float(np.dot(u, ref_u)) < 0:
            u *= -1.0
        if float(np.dot(v, ref_v)) < 0:
            v *= -1.0
        u_norm = float(np.linalg.norm(u))
        v_norm = float(np.linalg.norm(v))
        if u_norm <= 1e-8 or v_norm <= 1e-8:
            continue
        unit_us.append(_clamp_coronal_tilt(u / u_norm, tilt_limit_deg=tilt_limit_deg))
        unit_vs.append(_clamp_coronal_tilt(v / v_norm, tilt_limit_deg=tilt_limit_deg))
        u_norms.append(u_norm)
        v_norms.append(v_norm)
    if not unit_us or not unit_vs:
        return _CommonFrame(
            u_dir=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            v_dir=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            median_u_norm=456.0,
            median_v_norm=320.0,
            tilt_deg=0.0,
        )
    u_dir = _normalized(np.mean(np.stack(unit_us, axis=0), axis=0), fallback=ref_u)
    v_raw = np.mean(np.stack(unit_vs, axis=0), axis=0)
    v_raw = v_raw - (u_dir * float(np.dot(v_raw, u_dir)))
    v_dir = _normalized(v_raw, fallback=ref_v)
    u_dir, v_dir = _enforce_tilt_limit(u_dir, v_dir, tilt_limit_deg=tilt_limit_deg)
    return _CommonFrame(
        u_dir=u_dir,
        v_dir=v_dir,
        median_u_norm=float(np.median(u_norms)),
        median_v_norm=float(np.median(v_norms)),
        tilt_deg=_plane_tilt_deg(u_dir, v_dir),
    )


def _project_one_slice_to_common_frame(
    item: dict[str, object],
    *,
    common: _CommonFrame,
    hint: object | None,
    tilt_limit_deg: float,
    scale_range: tuple[float, float],
    max_y: int,
) -> tuple[dict[str, object], dict[str, object]]:
    values = np.asarray(item.get("anchoring", []), dtype=np.float64)
    if values.size < 9 or not np.all(np.isfinite(values[:9])):
        return dict(item), {"projection_status": "skipped_invalid_anchoring"}
    values = values[:9]
    center = values[0:3] + (0.5 * values[3:6]) + (0.5 * values[6:9])
    original_ap = _safe_estimate_ap(item)
    original_center_y = float(center[1])
    center[1] = _constrain_center_y(center[1], hint=hint, max_y=max_y)
    original_u_norm = float(np.linalg.norm(values[3:6]))
    original_v_norm = float(np.linalg.norm(values[6:9]))
    u_norm = float(np.clip(original_u_norm, common.median_u_norm * scale_range[0], common.median_u_norm * scale_range[1]))
    v_norm = float(np.clip(original_v_norm, common.median_v_norm * scale_range[0], common.median_v_norm * scale_range[1]))
    u_dir, v_dir = _enforce_tilt_limit(common.u_dir, common.v_dir, tilt_limit_deg=tilt_limit_deg)
    u = u_dir * u_norm
    v = v_dir * v_norm
    origin = center - (0.5 * u) - (0.5 * v)
    projected = dict(item)
    projected["anchoring"] = [float(value) for value in np.concatenate([origin, u, v])]
    projected_ap = _estimate_ap_from_anchoring(projected["anchoring"], projected)
    return projected, {
        "projection_status": "ok",
        "original_ap_mm": float(original_ap),
        "projected_ap_mm": float(projected_ap),
        "original_center_y": original_center_y,
        "projected_center_y": float(center[1]),
        "center_y_delta": float(center[1] - original_center_y),
        "original_u_norm": original_u_norm,
        "original_v_norm": original_v_norm,
        "projected_u_norm": u_norm,
        "projected_v_norm": v_norm,
        "projected_tilt_deg": float(_plane_tilt_deg(u, v)),
        "scale_factor_u": float(u_norm / max(original_u_norm, 1e-8)),
        "scale_factor_v": float(v_norm / max(original_v_norm, 1e-8)),
    }


def _search_lowdim_candidates(
    *,
    baseline_slice: dict[str, object],
    candidate_y_values: list[int],
    source: dict[str, np.ndarray],
    template_cache: CcfCoronalTemplateCache,
    scale_values: tuple[float, ...],
    roll_values: tuple[float, ...],
    prior_ap_mm: float,
    prior_width_mm: float,
    ap_prior_weight: float,
) -> dict[str, object]:
    values = np.asarray(baseline_slice.get("anchoring", []), dtype=np.float64)
    if values.size < 9:
        raise ValueError(f"Slice {baseline_slice.get('filename')!r} does not have a valid anchoring.")
    image_height = int(baseline_slice.get("height", source["source_mask"].shape[0]) or source["source_mask"].shape[0])
    image_width = int(baseline_slice.get("width", source["source_mask"].shape[1]) or source["source_mask"].shape[1])
    scored: list[dict[str, object]] = []
    for y_value in candidate_y_values:
        for roll_delta in roll_values:
            for scale_factor in scale_values:
                candidate = _anchoring_with_ap_roll_scale_local(
                    baseline=values,
                    target_y=float(y_value),
                    roll_delta_deg=float(roll_delta),
                    scale_factor=float(scale_factor),
                    image_width=image_width,
                    image_height=image_height,
                )
                label_map = _render_atlas_label_map_from_anchoring(
                    template_cache=template_cache,
                    anchoring=candidate,
                    output_shape=source["source_mask"].shape,
                )
                score = _score_rich_alignment(
                    image=source["image"],
                    source_mask=source["source_mask"],
                    atlas_label_map=label_map,
                )
                ap_mm = _estimate_ap_from_anchoring(candidate, baseline_slice)
                prior_penalty = _ap_prior_penalty(
                    ap_mm=ap_mm,
                    prior_ap_mm=prior_ap_mm,
                    prior_width_mm=prior_width_mm,
                    weight=ap_prior_weight,
                )
                score["total"] = float(score["total"] - prior_penalty)
                score["ap_prior_penalty"] = float(prior_penalty)
                scored.append(
                    {
                        "anchoring": candidate,
                        "quicknii_y": int(y_value),
                        "ap_mm": float(ap_mm),
                        "roll_delta_deg": float(roll_delta),
                        "scale_factor": float(scale_factor),
                        **score,
                    }
                )
    if not scored:
        raise RuntimeError("No low-dimensional candidates were generated.")
    scored.sort(key=lambda item: float(item["total"]), reverse=True)
    best = scored[0]
    top = [
        f"y={item['quicknii_y']}:ap={float(item['ap_mm']):.3f}:roll={float(item['roll_delta_deg']):.1f}:scale={float(item['scale_factor']):.3f}:score={float(item['total']):.4f}"
        for item in scored[:10]
    ]
    return {
        "anchoring": [float(value) for value in best["anchoring"]],
        "metadata": {
            "quicknii_y": int(best["quicknii_y"]),
            "ap_mm": float(best["ap_mm"]),
            "roll_delta_deg": float(best["roll_delta_deg"]),
            "scale_factor": float(best["scale_factor"]),
            "score_total": float(best["total"]),
            "score_mask_edge": float(best["mask_edge"]),
            "score_internal": float(best["internal"]),
            "score_ncc": float(best["ncc"]),
            "ap_prior_penalty": float(best["ap_prior_penalty"]),
            "candidate_y_count": len(candidate_y_values),
        },
        "row": {
            "quicknii_y": int(best["quicknii_y"]),
            "ap_mm": float(best["ap_mm"]),
            "roll_delta_deg": float(best["roll_delta_deg"]),
            "scale_factor": float(best["scale_factor"]),
            "score_total": float(best["total"]),
            "score_mask_edge": float(best["mask_edge"]),
            "score_internal": float(best["internal"]),
            "score_ncc": float(best["ncc"]),
            "ap_prior_penalty": float(best["ap_prior_penalty"]),
            "candidate_y_count": len(candidate_y_values),
            "top_candidates": ";".join(top),
        },
    }


def _search_intensity_landmark_candidates(
    *,
    baseline_slice: dict[str, object],
    candidate_y_values: list[int],
    source: dict[str, np.ndarray],
    template_cache: CcfCoronalTemplateCache,
    scale_values: tuple[float, ...],
    roll_values: tuple[float, ...],
    prior_ap_mm: float,
    prior_width_mm: float,
    ap_prior_weight: float,
) -> dict[str, object]:
    values = np.asarray(baseline_slice.get("anchoring", []), dtype=np.float64)
    if values.size < 9:
        raise ValueError(f"Slice {baseline_slice.get('filename')!r} does not have a valid anchoring.")
    image_height = int(baseline_slice.get("height", source["source_mask"].shape[0]) or source["source_mask"].shape[0])
    image_width = int(baseline_slice.get("width", source["source_mask"].shape[1]) or source["source_mask"].shape[1])

    baseline_label_map = _render_atlas_label_map_from_anchoring(
        template_cache=template_cache,
        anchoring=[float(value) for value in values[:9]],
        output_shape=source["source_mask"].shape,
    )
    baseline_score = _score_intensity_landmark_alignment(
        image=source["image"],
        source_mask=source["source_mask"],
        source_landmarks=source["source_landmarks"],
        atlas_label_map=baseline_label_map,
    )
    baseline_ap = _estimate_ap_from_anchoring([float(value) for value in values[:9]], baseline_slice)
    baseline_penalty = _ap_prior_penalty(
        ap_mm=baseline_ap,
        prior_ap_mm=prior_ap_mm,
        prior_width_mm=prior_width_mm,
        weight=ap_prior_weight,
    )
    baseline_total = float(baseline_score["total"] - baseline_penalty)

    scored: list[dict[str, object]] = []
    for y_value in candidate_y_values:
        for roll_delta in roll_values:
            for scale_factor in scale_values:
                candidate = _anchoring_with_ap_roll_scale_local(
                    baseline=values,
                    target_y=float(y_value),
                    roll_delta_deg=float(roll_delta),
                    scale_factor=float(scale_factor),
                    image_width=image_width,
                    image_height=image_height,
                )
                label_map = _render_atlas_label_map_from_anchoring(
                    template_cache=template_cache,
                    anchoring=candidate,
                    output_shape=source["source_mask"].shape,
                )
                score = _score_intensity_landmark_alignment(
                    image=source["image"],
                    source_mask=source["source_mask"],
                    source_landmarks=source["source_landmarks"],
                    atlas_label_map=label_map,
                )
                ap_mm = _estimate_ap_from_anchoring(candidate, baseline_slice)
                prior_penalty = _ap_prior_penalty(
                    ap_mm=ap_mm,
                    prior_ap_mm=prior_ap_mm,
                    prior_width_mm=prior_width_mm,
                    weight=ap_prior_weight,
                )
                score["total"] = float(score["total"] - prior_penalty)
                score["ap_prior_penalty"] = float(prior_penalty)
                scored.append(
                    {
                        "anchoring": candidate,
                        "quicknii_y": int(y_value),
                        "ap_mm": float(ap_mm),
                        "roll_delta_deg": float(roll_delta),
                        "scale_factor": float(scale_factor),
                        **score,
                    }
                )
    if not scored:
        raise RuntimeError("No intensity-landmark candidates were generated.")
    scored.sort(key=lambda item: float(item["total"]), reverse=True)
    best = scored[0]
    top = [
        (
            f"y={item['quicknii_y']}:ap={float(item['ap_mm']):.3f}:"
            f"roll={float(item['roll_delta_deg']):.1f}:scale={float(item['scale_factor']):.3f}:"
            f"score={float(item['total']):.4f}:ncc={float(item['atlas_intensity_ncc']):.4f}:"
            f"landmark={float(item['landmark']):.4f}:contourQC={float(item['contour_qc']):.4f}"
        )
        for item in scored[:10]
    ]
    return {
        "anchoring": [float(value) for value in best["anchoring"]],
        "metadata": {
            "quicknii_y": int(best["quicknii_y"]),
            "ap_mm": float(best["ap_mm"]),
            "roll_delta_deg": float(best["roll_delta_deg"]),
            "scale_factor": float(best["scale_factor"]),
            "score_total": float(best["total"]),
            "baseline_score_total": float(baseline_total),
            "score_gain_vs_baseline": float(best["total"] - baseline_total),
            "atlas_intensity_ncc": float(best["atlas_intensity_ncc"]),
            "landmark": float(best["landmark"]),
            "internal_chamfer": float(best["internal_chamfer"]),
            "contour_qc": float(best["contour_qc"]),
            "baseline_atlas_intensity_ncc": float(baseline_score["atlas_intensity_ncc"]),
            "baseline_landmark": float(baseline_score["landmark"]),
            "baseline_internal_chamfer": float(baseline_score["internal_chamfer"]),
            "baseline_contour_qc": float(baseline_score["contour_qc"]),
            "ap_prior_penalty": float(best["ap_prior_penalty"]),
            "candidate_y_count": len(candidate_y_values),
        },
        "row": {
            "quicknii_y": int(best["quicknii_y"]),
            "ap_mm": float(best["ap_mm"]),
            "roll_delta_deg": float(best["roll_delta_deg"]),
            "scale_factor": float(best["scale_factor"]),
            "score_total": float(best["total"]),
            "baseline_score_total": float(baseline_total),
            "score_gain_vs_baseline": float(best["total"] - baseline_total),
            "atlas_intensity_ncc": float(best["atlas_intensity_ncc"]),
            "landmark": float(best["landmark"]),
            "internal_chamfer": float(best["internal_chamfer"]),
            "contour_qc": float(best["contour_qc"]),
            "baseline_atlas_intensity_ncc": float(baseline_score["atlas_intensity_ncc"]),
            "baseline_landmark": float(baseline_score["landmark"]),
            "baseline_internal_chamfer": float(baseline_score["internal_chamfer"]),
            "baseline_contour_qc": float(baseline_score["contour_qc"]),
            "ap_prior_penalty": float(best["ap_prior_penalty"]),
            "candidate_y_count": len(candidate_y_values),
            "top_candidates": ";".join(top),
        },
    }


def _write_experiment_outputs(
    *,
    payload: dict[str, object],
    refined_slices: list[dict[str, object]],
    output_stem: Path,
    report_rows: list[dict[str, object]],
    report_suffix: str,
    teacher_json_path: Path | None,
    method: str,
    qc_dir: Path,
) -> ExperimentOutput:
    refined_slices.sort(key=lambda item: -_safe_estimate_ap(item))
    for nr, item in enumerate(refined_slices, start=1):
        item["nr"] = nr
    payload = dict(payload)
    payload["name"] = output_stem.stem
    payload["target"] = QDF1_EVO_TARGET
    payload["target-resolution"] = list(DEEPSLICE_MOUSE_QUICKNII_RESOLUTION)
    payload["slices"] = refined_slices
    json_out = output_stem.with_suffix(".json")
    csv_out = output_stem.with_suffix(".csv")
    xml_out = output_stem.with_suffix(".xml")
    write_prediction_bundle(payload=payload, json_path=json_out, csv_path=csv_out, xml_path=xml_out)
    report_path = output_stem.with_name(f"{output_stem.name}_{report_suffix}.csv")
    pd.DataFrame(report_rows).to_csv(report_path, index=False)
    teacher_report = None
    if teacher_json_path and Path(teacher_json_path).exists():
        teacher_report = output_stem.with_name(f"{output_stem.name}_teacher_comparison.csv")
        compare_prediction_to_teacher(
            prediction_json_path=json_out,
            teacher_json_path=Path(teacher_json_path),
            output_path=teacher_report,
        )
    return ExperimentOutput(
        method=method,
        json_path=json_out,
        csv_path=csv_out,
        xml_path=xml_out,
        report_path=report_path,
        qc_dir=qc_dir,
        teacher_report_path=teacher_report,
    )


def _score_and_qc(
    *,
    image_path: Path,
    anchoring: list[float],
    template_cache: CcfCoronalTemplateCache,
    qc_path: Path,
    preview_max_side: int,
) -> dict[str, dict[str, object]]:
    if not image_path.exists():
        return {
            "metadata": {"score_total": float("nan"), "qc_overlay": ""},
            "row": {"score_total": float("nan"), "qc_overlay": ""},
        }
    source = _prepare_source_for_scoring(image_path, preview_max_side=preview_max_side)
    label_map = _render_atlas_label_map_from_anchoring(
        template_cache=template_cache,
        anchoring=anchoring,
        output_shape=source["source_mask"].shape,
    )
    score = _score_rich_alignment(
        image=source["image"],
        source_mask=source["source_mask"],
        atlas_label_map=label_map,
    )
    write_rendered_qc_overlay(
        image_path=image_path,
        source_mask=source["source_mask_full"],
        anchoring=anchoring,
        template_cache=template_cache,
        output_path=qc_path,
    )
    return {
        "metadata": {
            "score_total": float(score["total"]),
            "score_mask_edge": float(score["mask_edge"]),
            "score_internal": float(score["internal"]),
            "score_ncc": float(score["ncc"]),
            "qc_overlay": str(qc_path),
        },
        "row": {
            "score_total": float(score["total"]),
            "score_mask_edge": float(score["mask_edge"]),
            "score_internal": float(score["internal"]),
            "score_ncc": float(score["ncc"]),
            "qc_overlay": str(qc_path),
        },
    }


def _load_teacher_lookup(teacher_json_path: Path | None) -> dict[str, dict[str, object]]:
    if not teacher_json_path or not Path(teacher_json_path).exists():
        return {}
    payload = json.loads(Path(teacher_json_path).read_text(encoding="utf-8"))
    return {
        normalize_slice_lookup_key(str(item.get("filename", ""))): item
        for item in payload.get("slices", [])
        if isinstance(item, dict)
    }


def _decide_selective_acceptance(
    *,
    baseline_slice: dict[str, object],
    candidate_slice: dict[str, object],
    teacher_slice: dict[str, object] | None,
    score_gain: float,
    min_objective_gain: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "objective_score_gain": float(score_gain),
        "min_objective_gain": float(min_objective_gain),
    }
    if teacher_slice is not None:
        baseline_metrics = _teacher_slice_metrics(baseline_slice, teacher_slice)
        candidate_metrics = _teacher_slice_metrics(candidate_slice, teacher_slice)
        row.update({f"baseline_{key}": value for key, value in baseline_metrics.items()})
        row.update({f"candidate_{key}": value for key, value in candidate_metrics.items()})
        improved = float(candidate_metrics["composite_error"]) < float(baseline_metrics["composite_error"]) - 1e-6
        return {
            "accepted": bool(improved),
            "reason": "teacher_metric_improved" if improved else "teacher_metric_not_improved",
            "row": row,
        }
    improved = math.isfinite(float(score_gain)) and float(score_gain) >= float(min_objective_gain)
    return {
        "accepted": bool(improved),
        "reason": "objective_score_improved_no_teacher" if improved else "objective_score_not_improved_no_teacher",
        "row": row,
    }


def _teacher_slice_metrics(slice_payload: dict[str, object], teacher_slice: dict[str, object]) -> dict[str, float]:
    pred_ap = _safe_estimate_ap(slice_payload)
    teacher_ap = _safe_estimate_ap(teacher_slice)
    pred_center = _anchoring_center(slice_payload)
    teacher_center = _anchoring_center(teacher_slice)
    center_delta_vox = float(np.linalg.norm(pred_center - teacher_center))
    center_delta_um = center_delta_vox * 25.0
    abs_ap_error_mm = abs(float(pred_ap) - float(teacher_ap))
    normal_angle = _normal_angle_delta_deg(slice_payload, teacher_slice)
    normal_component = float(normal_angle) * 25.0 if math.isfinite(float(normal_angle)) else 0.0
    composite = (abs_ap_error_mm * 1000.0) + center_delta_um + normal_component
    return {
        "abs_ap_error_mm": float(abs_ap_error_mm),
        "center_delta_um": float(center_delta_um),
        "normal_angle_delta_deg": float(normal_angle),
        "composite_error": float(composite),
    }


def _write_intensity_landmark_qc(
    *,
    image_path: Path,
    source: dict[str, np.ndarray],
    baseline_anchoring: list[float],
    candidate_anchoring: list[float],
    output_anchoring: list[float],
    template_cache: CcfCoronalTemplateCache,
    output_path: Path,
) -> None:
    if not image_path.exists():
        return
    image = source["image"]
    rgb = np.stack([image, image, image], axis=-1).astype(np.uint8, copy=True)
    baseline_labels = _render_atlas_label_map_from_anchoring(
        template_cache=template_cache,
        anchoring=baseline_anchoring,
        output_shape=image.shape,
    )
    candidate_labels = _render_atlas_label_map_from_anchoring(
        template_cache=template_cache,
        anchoring=candidate_anchoring,
        output_shape=image.shape,
    )
    output_labels = _render_atlas_label_map_from_anchoring(
        template_cache=template_cache,
        anchoring=output_anchoring,
        output_shape=image.shape,
    )
    _paint_mask_edges(rgb, source["source_mask"] > 0, color=(0, 190, 255))
    _paint_mask_edges(rgb, baseline_labels > 0, color=(80, 160, 255))
    _paint_mask_edges(rgb, candidate_labels > 0, color=(255, 230, 0))
    _paint_mask_edges(rgb, output_labels > 0, color=(80, 255, 120))
    pil_image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(pil_image)
    _draw_landmark_crosses(draw, source["source_landmarks"] > 0, color=(255, 40, 210), radius=3)
    _draw_landmark_crosses(draw, _atlas_landmark_map(candidate_labels) > 0, color=(255, 170, 0), radius=3)
    draw.rectangle((6, 6, 395, 78), fill=(0, 0, 0))
    draw.text((12, 10), "cyan: tissue edge / blue: baseline atlas", fill=(230, 240, 255))
    draw.text((12, 30), "yellow: best candidate / green: JSON output", fill=(230, 240, 255))
    draw.text((12, 50), "magenta: image landmarks / orange: atlas landmarks", fill=(230, 240, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(output_path, quality=84, optimize=True)


def _render_atlas_label_map_from_slice_with_markers(
    *,
    slice_payload: dict[str, object],
    output_shape: tuple[int, int],
    template_cache: CcfCoronalTemplateCache,
) -> np.ndarray:
    values = np.asarray(slice_payload.get("anchoring", []), dtype=np.float64)
    if values.size < 9:
        return np.zeros(output_shape[:2], dtype=np.uint32)
    height, width = int(output_shape[0]), int(output_shape[1])
    registration_width = int(slice_payload.get("width", width) or width)
    registration_height = int(slice_payload.get("height", height) or height)
    registration_slice = RegistrationSlice(
        filename=str(slice_payload.get("filename", "")),
        nr=slice_payload.get("nr") if isinstance(slice_payload.get("nr"), int) else None,
        width=registration_width,
        height=registration_height,
        origin=values[0:3],
        u=values[3:6],
        v=values[6:9],
        target_resolution=DEEPSLICE_MOUSE_QUICKNII_RESOLUTION,
        markers=[list(marker) for marker in slice_payload.get("markers", []) if isinstance(marker, list)],
        raw=slice_payload,
    )
    marker_warp = build_marker_inverse_warp(registration_slice, output_shape)
    labels = template_cache.labels
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    image_points = np.stack([xx, yy], axis=-1)
    registration_points = image_points_to_registration_source(
        image_points,
        output_shape,
        registration_slice,
        mapper=marker_warp,
        source_image_shape=output_shape,
    )
    xs = registration_points[..., 0] / max(float(registration_width), 1.0)
    ys = registration_points[..., 1] / max(float(registration_height), 1.0)
    coords = values[0:3] + (xs[..., None] * values[3:6]) + (ys[..., None] * values[6:9])
    rounded = np.rint(coords).astype(np.int32, copy=False)
    valid = (
        (rounded[..., 0] >= 0)
        & (rounded[..., 0] < labels.shape[0])
        & (rounded[..., 1] >= 0)
        & (rounded[..., 1] < labels.shape[1])
        & (rounded[..., 2] >= 0)
        & (rounded[..., 2] < labels.shape[2])
    )
    label_map = np.zeros((height, width), dtype=np.uint32)
    if np.any(valid):
        label_map[valid] = labels[
            rounded[..., 0][valid],
            rounded[..., 1][valid],
            rounded[..., 2][valid],
        ].astype(np.uint32, copy=False)
    return label_map


def _auto_priority_markers(
    *,
    source_mask: np.ndarray,
    atlas_mask: np.ndarray,
    atlas_label_map: np.ndarray,
    source_landmarks: np.ndarray,
    full_shape: tuple[int, int],
    max_markers: int,
    min_move_px: float,
    max_move_px: float,
    n_angles: int = 112,
) -> tuple[list[list[float]], list[str]]:
    """Place only the first-priority AtlasFitter pins requested for cleanup."""

    source = ndi.binary_fill_holes(source_mask > 0)
    atlas = ndi.binary_fill_holes(atlas_mask > 0)
    if int(source.sum()) < 64 or int(atlas.sum()) < 64:
        return [], []
    h, w = source.shape[:2]
    source_center = _mask_center_xy(source)
    atlas_center = _mask_center_xy(atlas)
    if source_center is None or atlas_center is None:
        return [], []
    scale_x = float(full_shape[1]) / max(float(w), 1.0)
    scale_y = float(full_shape[0]) / max(float(h), 1.0)
    max_move_px = max(float(max_move_px), 1.0)
    min_priority_separation = max(8.0, 0.030 * float(max(h, w)))

    selected: list[dict[str, object]] = []

    def add_marker(reason: str, atlas_xy: np.ndarray | None, target_xy: np.ndarray | None, score: float = 0.0) -> None:
        if len(selected) >= int(max_markers) or atlas_xy is None or target_xy is None:
            return
        ax, ay = float(atlas_xy[0]), float(atlas_xy[1])
        tx, ty = float(target_xy[0]), float(target_xy[1])
        if min(ax, ay, tx, ty) < 1.0 or ax > w - 2 or tx > w - 2 or ay > h - 2 or ty > h - 2:
            return
        move = float(np.hypot(tx - ax, ty - ay))
        if move > max_move_px:
            return
        if move < min(float(min_move_px), 0.75):
            return
        for existing in selected:
            marker = existing["marker"]
            ex = float(marker[0]) / scale_x
            ey = float(marker[1]) / scale_y
            etx = float(marker[2]) / scale_x
            ety = float(marker[3]) / scale_y
            if np.hypot(ax - ex, ay - ey) < min_priority_separation:
                return
            if np.hypot(tx - etx, ty - ety) < min_priority_separation:
                return
        selected.append(
            {
                "reason": reason,
                "score": float(score),
                "marker": [
                    float(ax * scale_x),
                    float(ay * scale_y),
                    float(tx * scale_x),
                    float(ty * scale_y),
                ],
            }
        )

    for reason, atlas_xy, target_xy in _midline_surface_markers(
        atlas=atlas,
        source=source,
        atlas_center=atlas_center,
        source_center=source_center,
    ):
        add_marker(reason, atlas_xy, target_xy, score=1000.0)

    for candidate in _matched_outer_concavity_candidates(
        atlas=atlas,
        source=source,
        atlas_center=atlas_center,
        source_center=source_center,
        n_angles=n_angles,
        max_count=max(0, int(max_markers) - len(selected)),
        max_move_px=max_move_px,
    ):
        add_marker(
            "atlas_outer_concavity",
            np.asarray(candidate["atlas_xy"], dtype=np.float64),
            np.asarray(candidate["target_xy"], dtype=np.float64),
            score=float(candidate["score"]),
        )

    for candidate in _ventricle_label_corner_candidates(
        atlas_label_map=atlas_label_map,
        source_landmarks=source_landmarks,
        atlas_mask=atlas,
        source_mask=source,
        max_count=max(0, int(max_markers) - len(selected)),
        max_move_px=max_move_px,
    ):
        add_marker(
            str(candidate["reason"]),
            np.asarray(candidate["atlas_xy"], dtype=np.float64),
            np.asarray(candidate["target_xy"], dtype=np.float64),
            score=float(candidate["score"]),
        )

    selected = selected[: int(max_markers)]
    return [list(row["marker"]) for row in selected], [str(row["reason"]) for row in selected]


def _mask_center_xy(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        return None
    return np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def _mask_principal_axes_xy(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask > 0)
    if xs.size < 3:
        return np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])
    points = np.column_stack((xs.astype(np.float64), ys.astype(np.float64)))
    points -= points.mean(axis=0, keepdims=True)
    cov = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    major = vectors[:, int(np.argmax(values))]
    minor = vectors[:, int(np.argmin(values))]
    if major[0] < 0:
        major = -major
    if minor[1] < 0:
        minor = -minor
    major = major / max(float(np.linalg.norm(major)), 1.0e-9)
    minor = minor / max(float(np.linalg.norm(minor)), 1.0e-9)
    return major.astype(np.float64), minor.astype(np.float64)


def _ray_boundary_xy(mask: np.ndarray, center_xy: np.ndarray, direction_xy: np.ndarray) -> np.ndarray | None:
    h, w = mask.shape[:2]
    direction = np.asarray(direction_xy, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-9:
        return None
    direction /= norm
    center = np.asarray(center_xy, dtype=np.float64)
    max_radius = int(np.ceil(np.hypot(max(center[0], w - center[0]), max(center[1], h - center[1])))) + 4
    steps = np.arange(0, max_radius + 1, dtype=np.float64)
    x_vals = np.rint(center[0] + (direction[0] * steps)).astype(np.int32)
    y_vals = np.rint(center[1] + (direction[1] * steps)).astype(np.int32)
    valid = (x_vals >= 0) & (x_vals < w) & (y_vals >= 0) & (y_vals < h)
    if not np.any(valid):
        return None
    x_vals = x_vals[valid]
    y_vals = y_vals[valid]
    hits = np.flatnonzero(mask[y_vals, x_vals] > 0)
    if hits.size == 0:
        return None
    index = int(hits[-1])
    return np.asarray([float(x_vals[index]), float(y_vals[index])], dtype=np.float64)


def _midline_surface_markers(
    *,
    atlas: np.ndarray,
    source: np.ndarray,
    atlas_center: np.ndarray,
    source_center: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    pairs: list[tuple[str, np.ndarray, np.ndarray]] = []
    dorsal_atlas = _central_surface_point(atlas, float(atlas_center[0]), side="dorsal")
    dorsal_source = _central_surface_point(source, float(source_center[0]), side="dorsal")
    if dorsal_atlas is not None and dorsal_source is not None:
        pairs.append(("ML0_dorsal", dorsal_atlas, dorsal_source))
    ventral_atlas = _central_surface_point(atlas, float(atlas_center[0]), side="ventral")
    ventral_source = _central_surface_point(source, float(source_center[0]), side="ventral")
    if ventral_atlas is not None and ventral_source is not None:
        pairs.append(("ML0_ventral", ventral_atlas, ventral_source))
    return pairs


def _central_surface_point(mask: np.ndarray, center_x: float, *, side: str) -> np.ndarray | None:
    mask_bool = mask > 0
    if int(mask_bool.sum()) < 32:
        return None
    x0, y0, x1, y1 = _mask_bbox(mask_bool)
    width = max(float(x1 - x0), 1.0)
    height = max(float(y1 - y0), 1.0)
    half_window = int(round(min(max(8.0, 0.055 * width), 0.16 * width)))
    start_x = max(int(round(center_x)) - half_window, 0)
    end_x = min(int(round(center_x)) + half_window + 1, mask_bool.shape[1])
    columns: list[tuple[int, int]] = []
    for x_value in range(start_x, end_x):
        ys = np.flatnonzero(mask_bool[:, x_value])
        if ys.size == 0:
            continue
        y_value = int(ys.min()) if side == "dorsal" else int(ys.max())
        columns.append((x_value, y_value))
    if not columns:
        return None
    xs = np.asarray([item[0] for item in columns], dtype=np.float64)
    ys = np.asarray([item[1] for item in columns], dtype=np.float64)
    closest_index = int(np.argmin(np.abs(xs - float(center_x))))
    if side == "dorsal":
        candidate_index = int(np.argmax(ys))
        depth = float(ys[candidate_index] - np.median(ys))
    else:
        candidate_index = int(np.argmin(ys))
        depth = float(np.median(ys) - ys[candidate_index])
    if depth < max(2.0, 0.010 * height):
        candidate_index = closest_index
    return np.asarray([float(xs[candidate_index]), float(ys[candidate_index])], dtype=np.float64)


def _matched_outer_concavity_candidates(
    *,
    atlas: np.ndarray,
    source: np.ndarray,
    atlas_center: np.ndarray,
    source_center: np.ndarray,
    n_angles: int,
    max_count: int,
    max_move_px: float,
) -> list[dict[str, object]]:
    if max_count <= 0:
        return []
    atlas_points, atlas_radii = _radial_boundary_profile(atlas, atlas_center, n_angles=n_angles)
    source_points, source_radii = _radial_boundary_profile(source, source_center, n_angles=n_angles)
    atlas_depth = _radial_concavity_depths(atlas_radii)
    source_depth = _radial_concavity_depths(source_radii)
    min_atlas_depth = max(5.0, 0.020 * float(max(atlas.shape[:2])))
    min_source_depth = max(4.0, 0.015 * float(max(source.shape[:2])))
    candidates: list[dict[str, object]] = []
    for angle_index in range(int(n_angles)):
        atlas_xy = atlas_points[angle_index]
        if atlas_xy is None or atlas_depth[angle_index] < min_atlas_depth:
            continue
        theta = (2.0 * math.pi * float(angle_index)) / float(n_angles)
        # Avoid duplicating the dorsal/ventral ML0 points; those are handled explicitly.
        if abs(math.cos(theta)) < 0.22:
            continue
        neighbor_indices = [((angle_index + delta) % int(n_angles)) for delta in range(-2, 3)]
        valid_source = [
            idx
            for idx in neighbor_indices
            if source_points[idx] is not None and source_depth[idx] >= min_source_depth
        ]
        if not valid_source:
            continue
        source_index = max(valid_source, key=lambda idx: float(source_depth[idx]))
        target_xy = source_points[source_index]
        if target_xy is None:
            continue
        move = float(np.linalg.norm(np.asarray(target_xy) - np.asarray(atlas_xy)))
        if move > float(max_move_px):
            continue
        score = float(atlas_depth[angle_index] + source_depth[source_index] - (0.2 * move))
        candidates.append({"score": score, "atlas_xy": atlas_xy, "target_xy": target_xy})
    candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    return candidates[: int(max_count)]


def _radial_boundary_profile(
    mask: np.ndarray,
    center_xy: np.ndarray,
    *,
    n_angles: int,
) -> tuple[list[np.ndarray | None], np.ndarray]:
    mask_bool = mask > 0
    points: list[np.ndarray | None] = [None] * int(n_angles)
    radii = np.full(int(n_angles), np.nan, dtype=np.float64)
    for angle_index in range(int(n_angles)):
        theta = (2.0 * math.pi * float(angle_index)) / float(n_angles)
        direction = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
        point = _ray_boundary_xy(mask_bool, center_xy, direction)
        if point is None:
            continue
        points[angle_index] = point
        radii[angle_index] = float(np.linalg.norm(point - center_xy))
    return points, radii


def _radial_concavity_depths(radii: np.ndarray) -> np.ndarray:
    depths = np.zeros(radii.shape, dtype=np.float64)
    count = int(radii.size)
    for index in range(count):
        radius = radii[index]
        if not np.isfinite(radius):
            continue
        neighbor_indices = [((index + delta) % count) for delta in range(-5, 6) if delta != 0]
        neighbor_values = radii[neighbor_indices]
        neighbor_values = neighbor_values[np.isfinite(neighbor_values)]
        if neighbor_values.size < 4:
            continue
        prev_radius = radii[(index - 1) % count]
        next_radius = radii[(index + 1) % count]
        if np.isfinite(prev_radius) and radius > prev_radius:
            continue
        if np.isfinite(next_radius) and radius > next_radius:
            continue
        depths[index] = max(0.0, float(np.nanmedian(neighbor_values) - radius))
    return depths


def _ventricle_label_corner_candidates(
    *,
    atlas_label_map: np.ndarray,
    source_landmarks: np.ndarray,
    atlas_mask: np.ndarray,
    source_mask: np.ndarray,
    max_count: int,
    max_move_px: float,
) -> list[dict[str, object]]:
    if max_count <= 0:
        return []
    ventricle_labels = (81, 129, 145)
    ventricle_mask = np.isin(atlas_label_map, ventricle_labels)
    if int(ventricle_mask.sum()) < 8:
        return []
    source_corner_points = _source_hole_corner_points(source_mask)
    if source_corner_points.size == 0:
        sy, sx = np.nonzero(source_landmarks > 0)
        source_corner_points = np.column_stack((sx.astype(np.float64), sy.astype(np.float64)))
    if source_corner_points.size == 0:
        return []
    labels, count = ndi.label(ventricle_mask)
    candidates: list[dict[str, object]] = []
    local_radius = min(float(max_move_px), max(10.0, 0.045 * float(max(source_mask.shape[:2]))))
    for label_index in range(1, int(count) + 1):
        component = labels == label_index
        area = int(component.sum())
        if area < 10:
            continue
        component_ids = np.asarray(atlas_label_map[component], dtype=np.int64)
        reason = "lateral_ventricle_corner" if int(np.count_nonzero(component_ids == 81)) >= int(np.count_nonzero(component_ids == 129)) else "third_ventricle_corner"
        for atlas_xy in _component_corner_points(component, max_points=4):
            distances = np.linalg.norm(source_corner_points - atlas_xy[None, :], axis=1)
            best_index = int(np.argmin(distances))
            distance = float(distances[best_index])
            if distance > local_radius:
                continue
            target_xy = source_corner_points[best_index]
            move = float(np.linalg.norm(target_xy - atlas_xy))
            if move > float(max_move_px):
                continue
            score = float((20.0 if reason == "third_ventricle_corner" else 12.0) - distance)
            candidates.append(
                {
                    "reason": reason,
                    "score": score,
                    "atlas_xy": atlas_xy,
                    "target_xy": target_xy,
                }
            )
    candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    limited: list[dict[str, object]] = []
    per_reason_count: dict[str, int] = {}
    total_limit = min(int(max_count), 3)
    for candidate in candidates:
        reason = str(candidate["reason"])
        if per_reason_count.get(reason, 0) >= 2:
            continue
        limited.append(candidate)
        per_reason_count[reason] = per_reason_count.get(reason, 0) + 1
        if len(limited) >= total_limit:
            break
    return limited


def _source_hole_corner_points(source_mask: np.ndarray) -> np.ndarray:
    source = source_mask > 0
    if int(source.sum()) < 64:
        return np.zeros((0, 2), dtype=np.float64)
    holes = ndi.binary_fill_holes(source) & ~source
    labels, count = ndi.label(holes)
    points: list[np.ndarray] = []
    source_area = max(int(source.sum()), 1)
    for label_index in range(1, int(count) + 1):
        component = labels == label_index
        area = int(component.sum())
        if area < 10 or area > int(0.20 * source_area):
            continue
        for point in _component_corner_points(component, max_points=4):
            points.append(point)
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def _component_corner_points(component: np.ndarray, *, max_points: int) -> list[np.ndarray]:
    ys, xs = np.nonzero(component > 0)
    if xs.size == 0:
        return []
    boundary = _binary_boundary(component > 0)
    by, bx = np.nonzero(boundary)
    if bx.size == 0:
        bx = xs
        by = ys
    boundary_points = np.column_stack((bx.astype(np.float64), by.astype(np.float64)))
    x0, y0, x1, y1 = _mask_bbox(component)
    bbox_targets = np.asarray(
        [
            [float(x0), float(y0)],
            [float(x1 - 1), float(y0)],
            [float(x0), float(y1 - 1)],
            [float(x1 - 1), float(y1 - 1)],
        ],
        dtype=np.float64,
    )
    points: list[np.ndarray] = []
    for target in bbox_targets:
        distances = np.linalg.norm(boundary_points - target[None, :], axis=1)
        index = int(np.argmin(distances))
        point = boundary_points[index].astype(np.float64, copy=True)
        if all(float(np.linalg.norm(point - existing)) >= 3.0 for existing in points):
            points.append(point)
        if len(points) >= int(max_points):
            break
    return points


def _atlas_concavity_candidates(
    *,
    atlas: np.ndarray,
    source: np.ndarray,
    atlas_center: np.ndarray,
    source_center: np.ndarray,
    n_angles: int,
    max_count: int,
) -> list[dict[str, object]]:
    if max_count <= 0:
        return []
    radii = np.full(int(n_angles), np.nan, dtype=np.float64)
    atlas_points: list[np.ndarray | None] = [None] * int(n_angles)
    target_points: list[np.ndarray | None] = [None] * int(n_angles)
    for angle_index in range(int(n_angles)):
        theta = (2.0 * math.pi * float(angle_index)) / float(n_angles)
        direction = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float64)
        atlas_xy = _ray_boundary_xy(atlas, atlas_center, direction)
        target_xy = _ray_boundary_xy(source, source_center, direction)
        if atlas_xy is None or target_xy is None:
            continue
        atlas_points[angle_index] = atlas_xy
        target_points[angle_index] = target_xy
        radii[angle_index] = float(np.linalg.norm(atlas_xy - atlas_center))
    candidates: list[dict[str, object]] = []
    min_depth = max(4.0, 0.020 * float(max(atlas.shape[:2])))
    for angle_index in range(int(n_angles)):
        radius = radii[angle_index]
        if not np.isfinite(radius):
            continue
        neighbor_indices = [((angle_index + delta) % int(n_angles)) for delta in range(-4, 5) if delta != 0]
        neighbor_values = radii[neighbor_indices]
        neighbor_values = neighbor_values[np.isfinite(neighbor_values)]
        if neighbor_values.size < 4:
            continue
        depth = float(np.nanmedian(neighbor_values) - radius)
        if depth < min_depth:
            continue
        prev_radius = radii[(angle_index - 1) % int(n_angles)]
        next_radius = radii[(angle_index + 1) % int(n_angles)]
        if np.isfinite(prev_radius) and radius > prev_radius:
            continue
        if np.isfinite(next_radius) and radius > next_radius:
            continue
        atlas_xy = atlas_points[angle_index]
        target_xy = target_points[angle_index]
        if atlas_xy is None or target_xy is None:
            continue
        candidates.append(
            {
                "score": depth + (0.2 * float(np.linalg.norm(target_xy - atlas_xy))),
                "atlas_xy": atlas_xy,
                "target_xy": target_xy,
            }
        )
    candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    return candidates[: int(max_count)]


def _ventricle_corner_candidates(
    *,
    atlas_label_map: np.ndarray,
    source_landmarks: np.ndarray,
    atlas_mask: np.ndarray,
    source_mask: np.ndarray,
    atlas_center: np.ndarray,
    source_center: np.ndarray,
    max_count: int,
    max_move_px: float,
) -> list[dict[str, object]]:
    if max_count <= 0:
        return []
    atlas_landmarks = _atlas_landmark_map(atlas_label_map, max_points=120)
    ay, ax = np.nonzero(atlas_landmarks > 0)
    sy, sx = np.nonzero(source_landmarks > 0)
    if ax.size == 0 or sx.size == 0:
        return []
    atlas_points = np.column_stack((ax.astype(np.float64), ay.astype(np.float64)))
    source_points = np.column_stack((sx.astype(np.float64), sy.astype(np.float64)))
    bbox = _mask_bbox(atlas_mask)
    if bbox is None:
        return []
    x0, y0, x1, y1 = bbox
    width = max(float(x1 - x0), 1.0)
    height = max(float(y1 - y0), 1.0)
    central = (
        (np.abs(atlas_points[:, 0] - atlas_center[0]) <= 0.28 * width)
        & (atlas_points[:, 1] >= float(y0) + (0.18 * height))
        & (atlas_points[:, 1] <= float(y0) + (0.78 * height))
    )
    if int(np.count_nonzero(central)) < 2:
        central = (
            (np.abs(atlas_points[:, 0] - atlas_center[0]) <= 0.42 * width)
            & (atlas_points[:, 1] >= float(y0) + (0.12 * height))
            & (atlas_points[:, 1] <= float(y0) + (0.86 * height))
        )
    atlas_points = atlas_points[central]
    if atlas_points.size == 0:
        return []
    offset = source_center - atlas_center
    candidates: list[dict[str, object]] = []
    for atlas_xy in atlas_points:
        predicted = atlas_xy + offset
        distances = np.linalg.norm(source_points - predicted[None, :], axis=1)
        best_index = int(np.argmin(distances))
        target_xy = source_points[best_index]
        move = float(np.linalg.norm(target_xy - atlas_xy))
        if move > float(max_move_px):
            continue
        centrality = 1.0 - min(float(abs(atlas_xy[0] - atlas_center[0]) / max(width * 0.5, 1.0)), 1.0)
        score = (2.0 * centrality) + (1.0 / (1.0 + float(distances[best_index])))
        candidates.append({"score": score, "atlas_xy": atlas_xy, "target_xy": target_xy})
    candidates.sort(key=lambda row: float(row["score"]), reverse=True)
    return candidates[: int(max_count)]


def _auto_contour_markers(
    *,
    source_mask: np.ndarray,
    atlas_mask: np.ndarray,
    full_shape: tuple[int, int],
    max_markers: int,
    min_move_px: float,
    max_move_px: float,
    n_angles: int = 112,
) -> list[list[float]]:
    source = ndi.binary_fill_holes(source_mask > 0)
    atlas = ndi.binary_fill_holes(atlas_mask > 0)
    if int(source.sum()) < 64 or int(atlas.sum()) < 64:
        return []
    h, w = source.shape[:2]
    ys, xs = np.nonzero(source)
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    max_radius = float(np.hypot(max(center_x, w - center_x), max(center_y, h - center_y))) + 2.0
    scale_x = float(full_shape[1]) / max(float(w), 1.0)
    scale_y = float(full_shape[0]) / max(float(h), 1.0)

    candidates: list[dict[str, object]] = []
    for angle_index in range(int(n_angles)):
        theta = (2.0 * math.pi * float(angle_index)) / float(n_angles)
        ray = np.linspace(0.0, max_radius, int(max_radius) + 1)
        x_vals = np.rint(center_x + (np.cos(theta) * ray)).astype(np.int32)
        y_vals = np.rint(center_y + (np.sin(theta) * ray)).astype(np.int32)
        valid = (x_vals >= 0) & (x_vals < w) & (y_vals >= 0) & (y_vals < h)
        if not np.any(valid):
            continue
        x_vals = x_vals[valid]
        y_vals = y_vals[valid]
        if x_vals.size < 6:
            continue
        source_hits = np.flatnonzero(source[y_vals, x_vals])
        atlas_hits = np.flatnonzero(atlas[y_vals, x_vals])
        if source_hits.size == 0 or atlas_hits.size == 0:
            continue
        source_idx = int(source_hits[-1])
        atlas_idx = int(atlas_hits[-1])
        sx = float(x_vals[source_idx])
        sy = float(y_vals[source_idx])
        ax = float(x_vals[atlas_idx])
        ay = float(y_vals[atlas_idx])
        move = float(np.hypot(sx - ax, sy - ay))
        if move < float(min_move_px) or move > float(max_move_px):
            continue
        if min(sx, sy, ax, ay) < 2 or sx > w - 3 or ax > w - 3 or sy > h - 3 or ay > h - 3:
            continue
        candidates.append(
            {
                "angle": float(theta),
                "move": move,
                "marker": [
                    float(ax * scale_x),
                    float(ay * scale_y),
                    float(sx * scale_x),
                    float(sy * scale_y),
                ],
            }
        )
    if not candidates:
        return []
    candidates.sort(key=lambda row: float(row["move"]), reverse=True)
    selected: list[dict[str, object]] = []
    min_angle_sep = (2.0 * math.pi) / max(float(max_markers) * 1.7, 1.0)
    for candidate in candidates:
        angle = float(candidate["angle"])
        if all(_angle_distance(angle, float(other["angle"])) >= min_angle_sep for other in selected):
            selected.append(candidate)
        if len(selected) >= int(max_markers):
            break
    selected.sort(key=lambda row: float(row["angle"]))
    return [list(row["marker"]) for row in selected]


def _angle_distance(a: float, b: float) -> float:
    diff = abs(float(a) - float(b)) % (2.0 * math.pi)
    return min(diff, (2.0 * math.pi) - diff)


def _write_auto_marker_qc(
    *,
    image: np.ndarray,
    source_mask: np.ndarray,
    baseline_mask: np.ndarray,
    warped_mask: np.ndarray,
    markers: list[list[float]],
    full_shape: tuple[int, int],
    accepted: bool,
    baseline_score: float,
    warped_score: float,
    output_path: Path,
) -> None:
    rgb = np.stack([image, image, image], axis=-1).astype(np.uint8, copy=True)
    _paint_mask_edges(rgb, source_mask > 0, color=(0, 210, 255))
    _paint_mask_edges(rgb, baseline_mask > 0, color=(70, 150, 255))
    _paint_mask_edges(rgb, warped_mask > 0, color=(80, 255, 120) if accepted else (255, 220, 0))
    pil_image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(pil_image)
    h, w = image.shape[:2]
    scale_x = float(w) / max(float(full_shape[1]), 1.0)
    scale_y = float(h) / max(float(full_shape[0]), 1.0)
    for marker in markers:
        if len(marker) < 4:
            continue
        sx = float(marker[0]) * scale_x
        sy = float(marker[1]) * scale_y
        tx = float(marker[2]) * scale_x
        ty = float(marker[3]) * scale_y
        draw.line((sx, sy, tx, ty), fill=(255, 90, 70), width=2)
        draw.ellipse((sx - 3, sy - 3, sx + 3, sy + 3), outline=(70, 150, 255), width=2)
        draw.ellipse((tx - 4, ty - 4, tx + 4, ty + 4), outline=(255, 240, 90), width=2)
    draw.rectangle((6, 6, 440, 82), fill=(0, 0, 0))
    draw.text((12, 10), "cyan: tissue contour / blue: baseline atlas", fill=(230, 240, 255))
    draw.text((12, 30), "green/yellow: marker-warped atlas after auto pins", fill=(230, 240, 255))
    draw.text((12, 50), f"accepted={accepted} contour {baseline_score:.4f} -> {warped_score:.4f}", fill=(230, 240, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(output_path, quality=85, optimize=True)


def _paint_mask_edges(rgb: np.ndarray, mask: np.ndarray, *, color: tuple[int, int, int]) -> None:
    edge = _binary_boundary(mask > 0)
    if not np.any(edge):
        return
    edge = ndi.binary_dilation(edge, iterations=1)
    rgb[edge] = np.asarray(color, dtype=np.uint8)


def _draw_landmark_crosses(
    draw: ImageDraw.ImageDraw,
    landmarks: np.ndarray,
    *,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    ys, xs = np.nonzero(landmarks > 0)
    for y_value, x_value in zip(ys.tolist(), xs.tolist(), strict=False):
        x = int(x_value)
        y = int(y_value)
        draw.line((x - radius, y, x + radius, y), fill=color, width=1)
        draw.line((x, y - radius, x, y + radius), fill=color, width=1)


def _prepare_source_for_scoring(image_path: Path, *, preview_max_side: int) -> dict[str, np.ndarray]:
    image_full = _read_grayscale_uint8(image_path)
    source_mask_full = tissue_mask_from_grayscale(image_full)
    image_preview, _scale = _downsample_grayscale(image_full, max_side=preview_max_side)
    source_mask = tissue_mask_from_grayscale(image_preview)
    source_mask = _keep_primary_components(source_mask)
    source_landmarks = _source_landmark_map(image=image_preview, source_mask=source_mask)
    return {
        "image": image_preview,
        "source_mask": source_mask.astype(np.uint8),
        "source_mask_full": source_mask_full.astype(np.uint8),
        "source_landmarks": source_landmarks.astype(np.uint8),
    }


def _render_atlas_label_map_from_anchoring(
    *,
    template_cache: CcfCoronalTemplateCache,
    anchoring: list[float],
    output_shape: tuple[int, int],
) -> np.ndarray:
    labels = template_cache.labels
    values = np.asarray(anchoring, dtype=np.float64)
    if values.size < 9:
        return np.zeros(output_shape[:2], dtype=np.uint32)
    h, w = int(output_shape[0]), int(output_shape[1])
    yy, xx = np.mgrid[0:h, 0:w]
    x_norm = xx.astype(np.float64) / max(float(w - 1), 1.0)
    y_norm = yy.astype(np.float64) / max(float(h - 1), 1.0)
    coords = values[0:3] + (x_norm[..., None] * values[3:6]) + (y_norm[..., None] * values[6:9])
    rounded = np.rint(coords).astype(np.int32, copy=False)
    valid = (
        (rounded[..., 0] >= 0)
        & (rounded[..., 0] < labels.shape[0])
        & (rounded[..., 1] >= 0)
        & (rounded[..., 1] < labels.shape[1])
        & (rounded[..., 2] >= 0)
        & (rounded[..., 2] < labels.shape[2])
    )
    label_map = np.zeros((h, w), dtype=np.uint32)
    if np.any(valid):
        label_map[valid] = labels[
            rounded[..., 0][valid],
            rounded[..., 1][valid],
            rounded[..., 2][valid],
        ].astype(np.uint32, copy=False)
    return label_map


def _score_rich_alignment(
    *,
    image: np.ndarray,
    source_mask: np.ndarray,
    atlas_label_map: np.ndarray,
) -> dict[str, float]:
    atlas_mask = (atlas_label_map > 0).astype(np.uint8)
    mask_edge = _score_rendered_overlay(
        source_mask=source_mask.astype(np.uint8),
        atlas_mask=atlas_mask,
        source_bbox=_mask_bbox(source_mask),
        atlas_bbox=_mask_bbox(atlas_mask),
    )
    source_internal = _source_internal_edges(image=image, source_mask=source_mask)
    atlas_internal = _label_boundaries(atlas_label_map)
    atlas_internal &= atlas_mask > 0
    internal = _chamfer_similarity(source_internal, atlas_internal)
    ncc = _masked_ncc(
        ndi.gaussian_filter(_normalize_image(image), sigma=4.0),
        ndi.gaussian_filter(atlas_mask.astype(np.float32), sigma=4.0),
        np.logical_or(source_mask > 0, atlas_mask > 0),
    )
    ncc01 = (float(ncc) + 1.0) * 0.5
    total = (0.55 * float(mask_edge)) + (0.25 * float(internal)) + (0.20 * ncc01)
    return {
        "total": float(total),
        "mask_edge": float(mask_edge),
        "internal": float(internal),
        "ncc": float(ncc01),
    }


def _score_intensity_landmark_alignment(
    *,
    image: np.ndarray,
    source_mask: np.ndarray,
    source_landmarks: np.ndarray,
    atlas_label_map: np.ndarray,
) -> dict[str, float]:
    atlas_mask = (atlas_label_map > 0).astype(np.uint8)
    contour_qc = _score_rendered_overlay(
        source_mask=source_mask.astype(np.uint8),
        atlas_mask=atlas_mask,
        source_bbox=_mask_bbox(source_mask),
        atlas_bbox=_mask_bbox(atlas_mask),
    )
    atlas_intensity_ncc = _atlas_region_intensity_ncc(
        image=image,
        source_mask=source_mask,
        atlas_label_map=atlas_label_map,
    )
    atlas_internal = _atlas_internal_boundaries(atlas_label_map)
    source_internal = _source_internal_edges(image=image, source_mask=source_mask)
    internal_chamfer = _chamfer_similarity(source_internal, atlas_internal)
    atlas_landmarks = _atlas_landmark_map(atlas_label_map)
    landmark = _landmark_similarity(source_landmarks > 0, atlas_landmarks > 0)
    total = (0.62 * float(atlas_intensity_ncc)) + (0.25 * float(landmark)) + (0.13 * float(internal_chamfer))
    return {
        "total": float(total),
        "atlas_intensity_ncc": float(atlas_intensity_ncc),
        "landmark": float(landmark),
        "internal_chamfer": float(internal_chamfer),
        "contour_qc": float(contour_qc),
    }


def _atlas_region_intensity_ncc(
    *,
    image: np.ndarray,
    source_mask: np.ndarray,
    atlas_label_map: np.ndarray,
    min_region_px: int = 18,
) -> float:
    image_n = ndi.gaussian_filter(_normalize_image(image), sigma=1.2)
    atlas_mask = atlas_label_map > 0
    valid = (source_mask > 0) & atlas_mask
    if int(valid.sum()) < 64:
        return 0.0
    labels = atlas_label_map[valid].astype(np.int64, copy=False)
    values = image_n[valid].astype(np.float64, copy=False)
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    if unique_labels.size < 2:
        return 0.0
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=values)
    global_mean = float(values.mean())
    means = np.full(unique_labels.shape, global_mean, dtype=np.float64)
    large = counts >= int(min_region_px)
    means[large] = sums[large] / np.maximum(counts[large], 1)
    model = np.full(image_n.shape, global_mean, dtype=np.float32)
    model[valid] = means[inverse].astype(np.float32, copy=False)
    model = ndi.gaussian_filter(model, sigma=1.0)
    ncc = _masked_ncc(image_n, model, valid)
    return float((ncc + 1.0) * 0.5)


def _source_landmark_map(*, image: np.ndarray, source_mask: np.ndarray, max_points: int = 140) -> np.ndarray:
    normalized = ndi.gaussian_filter(_normalize_image(image), sigma=1.0)
    internal_mask = source_mask > 0
    outer = _binary_boundary(internal_mask)
    internal_mask &= ~ndi.binary_dilation(outer, iterations=5)
    if int(internal_mask.sum()) < 64:
        return np.zeros(source_mask.shape, dtype=np.uint8)
    response = feature.corner_harris(normalized, sigma=1.6, k=0.04)
    response[~internal_mask] = 0.0
    coords = feature.corner_peaks(
        response,
        min_distance=7,
        threshold_rel=0.08,
        num_peaks=int(max_points),
        exclude_border=False,
    )
    out = np.zeros(source_mask.shape, dtype=np.uint8)
    if coords.size:
        out[coords[:, 0], coords[:, 1]] = 1
    return out


def _atlas_landmark_map(label_map: np.ndarray, max_points: int = 140) -> np.ndarray:
    internal = _atlas_internal_boundaries(label_map)
    if int(internal.sum()) < 16:
        return np.zeros(label_map.shape, dtype=np.uint8)
    boundary_image = ndi.gaussian_filter(internal.astype(np.float32), sigma=1.0)
    response = feature.corner_harris(boundary_image, sigma=1.2, k=0.04)
    response[~internal] = 0.0
    coords = feature.corner_peaks(
        response,
        min_distance=6,
        threshold_rel=0.04,
        num_peaks=int(max_points),
        exclude_border=False,
    )
    out = np.zeros(label_map.shape, dtype=np.uint8)
    if coords.size:
        out[coords[:, 0], coords[:, 1]] = 1
    return out


def _atlas_internal_boundaries(label_map: np.ndarray) -> np.ndarray:
    atlas_mask = label_map > 0
    boundary = _label_boundaries(label_map)
    outer = _binary_boundary(atlas_mask)
    outer_band = ndi.binary_dilation(outer, iterations=3)
    return boundary & atlas_mask & ~outer_band


def _landmark_similarity(source_landmarks: np.ndarray, atlas_landmarks: np.ndarray) -> float:
    if not np.any(source_landmarks) or not np.any(atlas_landmarks):
        return 0.0
    source_points = ndi.binary_dilation(source_landmarks > 0, iterations=1)
    atlas_points = ndi.binary_dilation(atlas_landmarks > 0, iterations=1)
    return _chamfer_similarity(source_points, atlas_points)


def _source_internal_edges(*, image: np.ndarray, source_mask: np.ndarray) -> np.ndarray:
    normalized = _normalize_image(image)
    edges = feature.canny(normalized, sigma=1.2)
    outer = _binary_boundary(source_mask > 0)
    outer_band = ndi.binary_dilation(outer, iterations=4)
    return edges & (source_mask > 0) & ~outer_band


def _label_boundaries(label_map: np.ndarray) -> np.ndarray:
    boundary = np.zeros(label_map.shape, dtype=bool)
    boundary[1:, :] |= label_map[1:, :] != label_map[:-1, :]
    boundary[:, 1:] |= label_map[:, 1:] != label_map[:, :-1]
    return boundary & (label_map > 0)


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask > 0
    eroded = ndi.binary_erosion(mask, iterations=1)
    return mask & ~eroded


def _chamfer_similarity(a_edges: np.ndarray, b_edges: np.ndarray) -> float:
    if not np.any(a_edges) or not np.any(b_edges):
        return 0.0
    a_dist = ndi.distance_transform_edt(~a_edges)
    b_dist = ndi.distance_transform_edt(~b_edges)
    norm = float(max(a_edges.shape[:2]))
    dist = 0.5 * (
        float(np.mean(b_dist[a_edges])) / max(norm, 1.0)
        + float(np.mean(a_dist[b_edges])) / max(norm, 1.0)
    )
    return float(1.0 - min(max(dist, 0.0), 1.0))


def _masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0
    if int(valid.sum()) < 16:
        valid = np.ones(a.shape, dtype=bool)
    av = a[valid].astype(np.float64, copy=False)
    bv = b[valid].astype(np.float64, copy=False)
    av -= float(av.mean())
    bv -= float(bv.mean())
    denom = float(np.sqrt(np.sum(av * av) * np.sum(bv * bv)))
    if denom <= 1e-8:
        return 0.0
    return float(np.sum(av * bv) / denom)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    low, high = np.percentile(image, [2.0, 99.5])
    if high <= low:
        return np.zeros(image.shape, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _keep_primary_components(mask: np.ndarray, *, min_fraction: float = 0.04) -> np.ndarray:
    labels, count = ndi.label(mask > 0)
    if count <= 1:
        return (mask > 0).astype(np.uint8)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    max_size = int(sizes.max(initial=0))
    keep_ids = np.flatnonzero(sizes >= max_size * float(min_fraction))
    if keep_ids.size == 0:
        keep_ids = np.asarray([int(np.argmax(sizes))])
    kept = np.isin(labels, keep_ids)
    return ndi.binary_fill_holes(kept).astype(np.uint8)


def _constrain_center_y(center_y: float, *, hint: object | None, max_y: int) -> float:
    if hint is None:
        return float(np.clip(center_y, 0, max_y))
    hint_ap = float(getattr(hint, "ap_mm"))
    tolerance = max(float(getattr(hint, "tolerance_mm")), 0.0)
    target_y = ap_mm_to_quicknii_y(hint_ap)
    half_width = (tolerance * 1000.0) / 25.0
    return float(np.clip(center_y, max(0.0, target_y - half_width), min(float(max_y), target_y + half_width)))


def _clamp_coronal_tilt(vector: np.ndarray, *, tilt_limit_deg: float) -> np.ndarray:
    out = np.asarray(vector, dtype=np.float64).copy()
    horizontal = float(np.linalg.norm(out[[0, 2]]))
    if horizontal <= 1e-8:
        return _normalized(out, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
    max_y = math.tan(math.radians(float(tilt_limit_deg))) * horizontal
    out[1] = float(np.clip(out[1], -max_y, max_y))
    return _normalized(out, fallback=vector)


def _enforce_tilt_limit(
    u_dir: np.ndarray,
    v_dir: np.ndarray,
    *,
    tilt_limit_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    u = _clamp_coronal_tilt(u_dir, tilt_limit_deg=tilt_limit_deg)
    v = _clamp_coronal_tilt(v_dir, tilt_limit_deg=tilt_limit_deg)
    for _ in range(8):
        v = v - (u * float(np.dot(v, u)))
        v = _normalized(v, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float64))
        tilt = _plane_tilt_deg(u, v)
        if tilt <= float(tilt_limit_deg) + 1e-6:
            return u, v
        u[1] *= 0.75
        v[1] *= 0.75
        u = _normalized(u, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
        v = _normalized(v, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float64))
    return u, v


def _plane_tilt_deg(u: np.ndarray, v: np.ndarray) -> float:
    normal = np.cross(u, v)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-8:
        return 90.0
    normal = normal / norm
    ap_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cosine = min(max(abs(float(np.dot(normal, ap_axis))), -1.0), 1.0)
    return float(math.degrees(math.acos(cosine)))


def _normalized(vector: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        return vector.astype(np.float64, copy=False) / norm
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm > 1e-8:
        return fallback.astype(np.float64, copy=False) / fallback_norm
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def _sample_values(values: list[int], *, max_count: int) -> list[int]:
    if len(values) <= max_count:
        return list(values)
    indices = np.linspace(0, len(values) - 1, num=max_count)
    sampled = {int(values[int(round(index))]) for index in indices}
    sampled.add(int(values[len(values) // 2]))
    return sorted(sampled)


def _anchoring_center_y(anchoring: list[float]) -> float:
    values = np.asarray(anchoring, dtype=np.float64)
    if values.size < 9:
        return float("nan")
    return float(values[1] + (0.5 * values[4]) + (0.5 * values[7]))


def _local_candidate_y_values(
    *,
    allowed_values: list[int],
    center_y: float,
    half_width_vox: int,
    step_vox: int,
) -> list[int]:
    allowed = sorted({int(value) for value in allowed_values})
    if not allowed:
        return []
    if not math.isfinite(float(center_y)):
        return _sample_values(allowed, max_count=9)
    half_width = max(int(half_width_vox), 0)
    step = max(int(step_vox), 1)
    allowed_set = set(allowed)
    center_int = int(round(float(center_y)))
    candidates = {min(allowed, key=lambda value: abs(value - center_int))}
    for offset in range(-half_width, half_width + 1, step):
        value = int(round(center_int + offset))
        value = min(max(value, allowed[0]), allowed[-1])
        if value in allowed_set:
            candidates.add(value)
        else:
            candidates.add(min(allowed, key=lambda allowed_value: abs(allowed_value - value)))
    return sorted(candidates)


def _anchoring_with_ap_roll_scale_local(
    *,
    baseline: np.ndarray,
    target_y: float,
    roll_delta_deg: float,
    scale_factor: float,
    image_width: int,
    image_height: int,
) -> list[float]:
    origin = baseline[0:3].astype(np.float64, copy=True)
    u = baseline[3:6].astype(np.float64, copy=True)
    v = baseline[6:9].astype(np.float64, copy=True)
    center = origin + (0.5 * u) + (0.5 * v)
    center[1] = float(target_y)
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    theta = math.radians(float(roll_delta_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    scaled_u = u * float(scale_factor)
    scaled_v = v * float(scale_factor)
    u_rot = (cos_t * scaled_u) + (sin_t * scaled_v * (width / height))
    v_rot = (-sin_t * scaled_u * (height / width)) + (cos_t * scaled_v)
    origin_new = center - (0.5 * u_rot) - (0.5 * v_rot)
    return [float(value) for value in np.concatenate([origin_new, u_rot, v_rot])]


def _ap_prior_penalty(*, ap_mm: float, prior_ap_mm: float, prior_width_mm: float, weight: float) -> float:
    if not math.isfinite(float(ap_mm)):
        return 0.0
    width = max(float(prior_width_mm), 0.025)
    return float(weight) * min(abs(float(ap_mm) - float(prior_ap_mm)) / width, 1.0)
