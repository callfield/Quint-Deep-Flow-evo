"""Atlas registration rasterization and overlay rendering."""

from __future__ import annotations

import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
import tifffile

try:
    from numba import njit, prange
except Exception:  # pragma: no cover - optional acceleration
    njit = None
    prange = range

from atlas.display_codes import midline_offset, stable_region_display_code
from atlas.repository import AtlasRepository
from data_models.models import DetectedObject, RegistrationSlice
from io_utils.image_io import ensure_rgb, grayscale_intensity
from registration.nonlinear import build_marker_inverse_warp, build_piecewise_affine_mapper, image_points_to_registration_source

CELL_OUTLINE_RGB = (235, 235, 235)
REGION_OUTLINE_RGB = (230, 220, 120)
FALLBACK_CHANNEL_COLORS = (
    (90, 220, 255),
    (255, 110, 110),
    (130, 255, 150),
    (255, 195, 90),
)
CHANNEL_PATTERN = re.compile(r"CH(\d+)", re.IGNORECASE)
QDF2D_TRANSFORM_KEY = "qdf2d_transform"


if njit is not None:

    @njit(cache=True)
    def _best_replacement_values_numba(sorted_neighbors: np.ndarray, current_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_rows, n_neighbors = sorted_neighbors.shape
        best_values = np.zeros(n_rows, dtype=np.uint32)
        best_counts = np.zeros(n_rows, dtype=np.int16)
        current_counts = np.zeros(n_rows, dtype=np.int16)
        for row in range(n_rows):
            best_val = 0
            best_count = 0
            current_val = np.uint32(current_values[row])
            current_count = 0
            run_count = 0
            prev_val = np.uint32(0)
            for col in range(n_neighbors):
                value = np.uint32(sorted_neighbors[row, col])
                if value == 0:
                    run_count = 0
                else:
                    if col > 0 and value == prev_val:
                        run_count += 1
                    else:
                        run_count = 1
                    if run_count > best_count:
                        best_count = run_count
                        best_val = value
                    if value == current_val:
                        current_count += 1
                prev_val = value
            best_values[row] = best_val
            best_counts[row] = np.int16(best_count)
            current_counts[row] = np.int16(current_count)
        return best_values, best_counts, current_counts
else:
    _best_replacement_values_numba = None


if njit is not None:

    @njit(cache=True)
    def _round_half_to_even_numba(value: float) -> int:
        lower = np.floor(value)
        fraction = value - lower
        if fraction < 0.5:
            return int(lower)
        if fraction > 0.5:
            return int(lower + 1.0)
        lower_int = int(lower)
        if lower_int % 2 == 0:
            return lower_int
        return int(lower + 1.0)

    @njit(cache=True, parallel=True)
    def _warp_qdf2d_maps_numba(
        region_map: np.ndarray,
        hemisphere_map: np.ndarray,
        minx: np.ndarray,
        miny: np.ndarray,
        maxx: np.ndarray,
        maxy: np.ndarray,
        target_ax: np.ndarray,
        target_ay: np.ndarray,
        decomp: np.ndarray,
        source_ax: np.ndarray,
        source_ay: np.ndarray,
        source_bdx: np.ndarray,
        source_bdy: np.ndarray,
        source_cdx: np.ndarray,
        source_cdy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        height, width = region_map.shape
        transformed_region = np.zeros(region_map.shape, dtype=np.uint32)
        transformed_hemi = np.zeros(hemisphere_map.shape, dtype=np.int8)
        n_triangles = minx.shape[0]
        for y in prange(height):
            for x in range(width):
                sx = float(x)
                sy = float(y)
                for tri in range(n_triangles):
                    xf = float(x)
                    yf = float(y)
                    if xf < minx[tri] or xf > maxx[tri] or yf < miny[tri] or yf > maxy[tri]:
                        continue
                    dx = xf - target_ax[tri]
                    dy = yf - target_ay[tri]
                    u = dx * decomp[tri, 0, 0] + dy * decomp[tri, 0, 1]
                    v = dx * decomp[tri, 1, 0] + dy * decomp[tri, 1, 1]
                    if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0 or (u + v) > 1.0:
                        continue
                    sx = source_ax[tri] + source_bdx[tri] * u + source_cdx[tri] * v
                    sy = source_ay[tri] + source_bdy[tri] * u + source_cdy[tri] * v
                    break
                xi = _round_half_to_even_numba(sx)
                yi = _round_half_to_even_numba(sy)
                if 0 <= xi < width and 0 <= yi < height:
                    transformed_region[y, x] = np.uint32(region_map[yi, xi])
                    transformed_hemi[y, x] = np.int8(hemisphere_map[yi, xi])
        return transformed_region, transformed_hemi
else:
    _warp_qdf2d_maps_numba = None


def qdf2d_transform_payload(registration_slice: RegistrationSlice) -> dict[str, object] | None:
    """Return QDF evo 2D-layer metadata when present on a registration slice."""

    raw = registration_slice.raw if isinstance(registration_slice.raw, dict) else {}
    payload = raw.get(QDF2D_TRANSFORM_KEY)
    if not isinstance(payload, dict) or bool(payload.get("_disable")):
        return None
    base_anchoring = payload.get("base_anchoring")
    if not isinstance(base_anchoring, list) or len(base_anchoring) < 9:
        return None
    return payload


def _base_registration_slice_from_qdf2d(
    registration_slice: RegistrationSlice,
    payload: dict[str, object],
) -> RegistrationSlice:
    base_anchoring = np.asarray(payload.get("base_anchoring", [])[:9], dtype=np.float64)
    if base_anchoring.size < 9:
        base_anchoring = np.concatenate(
            [
                np.asarray(registration_slice.origin, dtype=np.float64),
                np.asarray(registration_slice.u, dtype=np.float64),
                np.asarray(registration_slice.v, dtype=np.float64),
            ]
        )
    raw = dict(registration_slice.raw or {})
    raw[QDF2D_TRANSFORM_KEY] = {"_disable": True}
    return RegistrationSlice(
        filename=registration_slice.filename,
        nr=registration_slice.nr,
        width=registration_slice.width,
        height=registration_slice.height,
        origin=base_anchoring[0:3].astype(np.float64, copy=False),
        u=base_anchoring[3:6].astype(np.float64, copy=False),
        v=base_anchoring[6:9].astype(np.float64, copy=False),
        target_resolution=registration_slice.target_resolution,
        markers=[],
        raw=raw,
    )


def qdf2d_base_registration_slice(
    registration_slice: RegistrationSlice,
    payload: dict[str, object],
) -> RegistrationSlice:
    """Return the 3D base-reslice registration slice for a QDF evo 2D-layer slice."""

    return _base_registration_slice_from_qdf2d(registration_slice, payload)


def _scaled_marker_rows_for_output(
    marker_rows: list[list[float]],
    registration_size: tuple[int, int],
    output_shape: tuple[int, int],
    registration_offset_px: tuple[float, float] = (0.0, 0.0),
    registration_source_shape: tuple[int, int] | None = None,
) -> list[list[float]]:
    registration_height, registration_width = registration_size
    source_height, source_width = registration_source_shape or output_shape
    scale_x = float(source_width) / float(max(registration_width, 1))
    scale_y = float(source_height) / float(max(registration_height, 1))
    offset_x = float(registration_offset_px[0])
    offset_y = float(registration_offset_px[1])
    scaled: list[list[float]] = []
    for row in marker_rows:
        if len(row) < 4:
            continue
        scaled.append(
            [
                offset_x + (float(row[0]) * scale_x),
                offset_y + (float(row[1]) * scale_y),
                offset_x + (float(row[2]) * scale_x),
                offset_y + (float(row[3]) * scale_y),
            ]
        )
    return scaled


def qdf2d_marker_rows_for_output(
    registration_slice: RegistrationSlice,
    payload: dict[str, object],
    output_shape: tuple[int, int],
    registration_offset_px: tuple[float, float] = (0.0, 0.0),
    registration_source_shape: tuple[int, int] | None = None,
) -> list[list[float]]:
    """Build output-space marker rows for base-map to final-map 2D warping."""

    registration_height = max(int(registration_slice.height or output_shape[0]), 1)
    registration_width = max(int(registration_slice.width or output_shape[1]), 1)
    corner_rows_raw = payload.get("corner_markers")
    corner_rows = corner_rows_raw if isinstance(corner_rows_raw, list) else []
    rows: list[list[float]] = []
    rows.extend(
        _scaled_marker_rows_for_output(
            corner_rows,
            (registration_height, registration_width),
            output_shape,
            registration_offset_px=registration_offset_px,
            registration_source_shape=registration_source_shape,
        )
    )
    rows.extend(
        _scaled_marker_rows_for_output(
            registration_slice.markers,
            (registration_height, registration_width),
            output_shape,
            registration_offset_px=registration_offset_px,
            registration_source_shape=registration_source_shape,
        )
    )
    return rows


def apply_qdf2d_transform_to_maps(
    region_map: np.ndarray,
    hemisphere_map: np.ndarray,
    registration_slice: RegistrationSlice,
    payload: dict[str, object],
    registration_offset_px: tuple[float, float] = (0.0, 0.0),
    registration_source_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Apply the saved 2D layer transform to already-resliced base atlas maps."""

    output_shape = tuple(int(value) for value in region_map.shape[:2])
    height, width = output_shape
    marker_rows = qdf2d_marker_rows_for_output(
        registration_slice,
        payload,
        output_shape,
        registration_offset_px=registration_offset_px,
        registration_source_shape=registration_source_shape,
    )
    mapper = build_piecewise_affine_mapper(width, height, marker_rows)
    metrics = {
        "qdf2d_layer_enabled": 1.0,
        "qdf2d_marker_count_used": float(len(marker_rows)),
        "qdf2d_user_marker_count_used": float(len(registration_slice.markers)),
    }
    if mapper is None:
        return region_map, hemisphere_map, metrics

    use_numba_warp = os.environ.get("QDF_QDF2D_NUMBA_WARP", "0").strip().lower() in {"1", "true", "yes", "on"}
    if use_numba_warp and _warp_qdf2d_maps_numba is not None:
        triangles = [triangle for triangle in mapper.triangles if triangle.decomp is not None]
        if triangles:
            minx = np.asarray([triangle.minx for triangle in triangles], dtype=np.float64)
            miny = np.asarray([triangle.miny for triangle in triangles], dtype=np.float64)
            maxx = np.asarray([triangle.maxx for triangle in triangles], dtype=np.float64)
            maxy = np.asarray([triangle.maxy for triangle in triangles], dtype=np.float64)
            target_ax = np.asarray([triangle.target_a[0] for triangle in triangles], dtype=np.float64)
            target_ay = np.asarray([triangle.target_a[1] for triangle in triangles], dtype=np.float64)
            decomp = np.asarray([triangle.decomp for triangle in triangles], dtype=np.float64)
            source_ax = np.asarray([triangle.source_a[0] for triangle in triangles], dtype=np.float64)
            source_ay = np.asarray([triangle.source_a[1] for triangle in triangles], dtype=np.float64)
            source_bdx = np.asarray([triangle.source_b_delta[0] for triangle in triangles], dtype=np.float64)
            source_bdy = np.asarray([triangle.source_b_delta[1] for triangle in triangles], dtype=np.float64)
            source_cdx = np.asarray([triangle.source_c_delta[0] for triangle in triangles], dtype=np.float64)
            source_cdy = np.asarray([triangle.source_c_delta[1] for triangle in triangles], dtype=np.float64)
            transformed_region, transformed_hemi = _warp_qdf2d_maps_numba(
                region_map.astype(np.uint32, copy=False),
                hemisphere_map.astype(np.int8, copy=False),
                minx,
                miny,
                maxx,
                maxy,
                target_ax,
                target_ay,
                decomp,
                source_ax,
                source_ay,
                source_bdx,
                source_bdy,
                source_cdx,
                source_cdy,
            )
            metrics["qdf2d_numba_warp"] = 1.0
            return transformed_region, transformed_hemi, metrics

    transformed_region = np.zeros_like(region_map, dtype=np.uint32)
    transformed_hemi = np.zeros_like(hemisphere_map, dtype=np.int8)
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.float64),
        np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    target_points = np.stack([xx, yy], axis=-1)
    source_points = mapper.map_target_to_source(target_points)
    xi = np.rint(source_points[..., 0]).astype(np.int32)
    yi = np.rint(source_points[..., 1]).astype(np.int32)
    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    if np.any(valid):
        transformed_region[valid] = region_map[yi[valid], xi[valid]].astype(np.uint32, copy=False)
        transformed_hemi[valid] = hemisphere_map[yi[valid], xi[valid]].astype(np.int8, copy=False)
    return transformed_region, transformed_hemi, metrics


def _qdf2d_area_scale(payload: dict[str, object]) -> float:
    base_display = payload.get("base_display")
    display = payload.get("display")
    if not isinstance(base_display, dict) or not isinstance(display, dict):
        return 1.0
    try:
        base_span_ml = max(float(base_display.get("span_ml", 1.0)), 1e-6)
        base_span_dv = max(float(base_display.get("span_dv", 1.0)), 1e-6)
        span_ml = max(float(display.get("span_ml", base_span_ml)), 1e-6)
        span_dv = max(float(display.get("span_dv", base_span_dv)), 1e-6)
    except (TypeError, ValueError):
        return 1.0
    return max((base_span_ml / span_ml) * (base_span_dv / span_dv), 1e-6)


def build_registered_maps(
    atlas: AtlasRepository,
    registration_slice: RegistrationSlice,
    output_shape: tuple[int, int],
    midline_threshold_um: float,
    registration_offset_px: tuple[float, float] = (0.0, 0.0),
    registration_source_shape: tuple[int, int] | None = None,
    chunk_rows: int = 256,
    smooth_regions: bool = False,
    smoothing_kernel_size: int = 5,
    smoothing_iterations: int = 1,
    smoothing_downsample_factor: int = 1,
    simplify_contours: bool = False,
    contour_tolerance_px: float = 2.5,
    contour_min_component_area_px: int = 128,
    atlas_sampling_mode: str = "nearest",
    atlas_sampling_radius_vox: int = 2,
    atlas_sampling_batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Rasterize atlas labels and hemisphere assignments into image space."""

    atlas.load()
    qdf2d_payload = qdf2d_transform_payload(registration_slice)
    if qdf2d_payload is not None:
        base_slice = _base_registration_slice_from_qdf2d(registration_slice, qdf2d_payload)
        base_region_map, base_hemisphere_map, base_metrics = build_registered_maps(
            atlas,
            base_slice,
            output_shape,
            midline_threshold_um=midline_threshold_um,
            registration_offset_px=registration_offset_px,
            registration_source_shape=registration_source_shape,
            chunk_rows=chunk_rows,
            smooth_regions=smooth_regions,
            smoothing_kernel_size=smoothing_kernel_size,
            smoothing_iterations=smoothing_iterations,
            smoothing_downsample_factor=smoothing_downsample_factor,
            simplify_contours=simplify_contours,
            contour_tolerance_px=contour_tolerance_px,
            contour_min_component_area_px=contour_min_component_area_px,
            atlas_sampling_mode=atlas_sampling_mode,
            atlas_sampling_radius_vox=atlas_sampling_radius_vox,
            atlas_sampling_batch_size=atlas_sampling_batch_size,
        )
        region_map, hemisphere_map, qdf2d_metrics = apply_qdf2d_transform_to_maps(
            base_region_map,
            base_hemisphere_map,
            registration_slice,
            qdf2d_payload,
            registration_offset_px=registration_offset_px,
            registration_source_shape=registration_source_shape,
        )
        metrics = dict(base_metrics)
        metrics.update(qdf2d_metrics)
        area_scale = _qdf2d_area_scale(qdf2d_payload)
        if "pixel_area_um2" in metrics:
            metrics["pixel_area_um2"] = float(metrics["pixel_area_um2"]) / area_scale
        metrics["qdf2d_area_scale"] = float(area_scale)
        metrics["atlas_coverage_fraction"] = float((region_map > 0).sum() / max(region_map.size, 1))
        metrics["marker_warp_enabled"] = bool(registration_slice.markers)
        metrics["marker_count_used"] = int(len(registration_slice.markers))
        return region_map, hemisphere_map, metrics

    labels = atlas.require_labels()
    height, width = output_shape
    registration_height = max(int(registration_slice.height or height), 1)
    registration_width = max(int(registration_slice.width or width), 1)
    marker_warp = build_marker_inverse_warp(registration_slice, output_shape)
    region_map = np.zeros((height, width), dtype=np.uint32)
    hemisphere_map = np.zeros((height, width), dtype=np.int8)
    in_bounds_total = 0

    for row_start in range(0, height, chunk_rows):
        row_end = min(row_start + chunk_rows, height)
        yy, xx = np.meshgrid(
            np.arange(row_start, row_end, dtype=np.float64),
            np.arange(width, dtype=np.float64),
            indexing="ij",
        )
        image_points = np.stack([xx, yy], axis=-1)
        registration_points = image_points_to_registration_source(
            image_points,
            output_shape,
            registration_slice,
            mapper=marker_warp,
            image_offset_px=registration_offset_px,
            source_image_shape=registration_source_shape,
        )
        xs = registration_points[..., 0] / registration_width
        ys = registration_points[..., 1] / registration_height
        xq = registration_slice.origin[0] + (xs * registration_slice.u[0]) + (ys * registration_slice.v[0])
        yq = registration_slice.origin[1] + (xs * registration_slice.u[1]) + (ys * registration_slice.v[1])
        zq = registration_slice.origin[2] + (xs * registration_slice.u[2]) + (ys * registration_slice.v[2])

        xi = np.rint(xq).astype(np.int32)
        yi = np.rint(yq).astype(np.int32)
        zi = np.rint(zq).astype(np.int32)
        valid = (
            (xq >= 0.0)
            & (xq < labels.shape[0])
            & (yq >= 0.0)
            & (yq < labels.shape[1])
            & (zq >= 0.0)
            & (zq < labels.shape[2])
        )
        sampled = np.zeros((row_end - row_start, width), dtype=np.uint32)
        if np.any(valid):
            sample_points = np.stack([xq, yq, zq], axis=-1)
            sampled[valid] = atlas.sample_labels_with_mode(
                sample_points[valid],
                mode=atlas_sampling_mode,
                radius_vox=atlas_sampling_radius_vox,
                batch_size=atlas_sampling_batch_size,
            ).astype(np.uint32, copy=False)
        region_map[row_start:row_end] = sampled

        ml_um = (xq * atlas.voxel_size_um) - atlas.config.allen_bregma_um[2]
        hemi_chunk = np.zeros_like(sampled, dtype=np.int8)
        tissue_mask = sampled > 0
        hemi_chunk[tissue_mask & (ml_um >= 0.0)] = 1
        hemi_chunk[tissue_mask & (ml_um < 0.0)] = -1
        hemisphere_map[row_start:row_end] = hemi_chunk
        in_bounds_total += int(valid.sum())

    raw_coverage_fraction = float((region_map > 0).sum() / max(region_map.size, 1))
    if smooth_regions and smoothing_kernel_size >= 3 and smoothing_iterations > 0:
        region_map = _smooth_label_map(
            region_map,
            kernel_size=smoothing_kernel_size,
            iterations=smoothing_iterations,
            chunk_rows=chunk_rows,
            downsample_factor=smoothing_downsample_factor,
        )
    if simplify_contours and contour_tolerance_px > 0:
        region_map = _simplify_label_map_contours(
            region_map,
            tolerance_px=contour_tolerance_px,
            min_component_area_px=contour_min_component_area_px,
        )

    metrics = {
        "atlas_coverage_fraction": float((region_map > 0).sum() / max(region_map.size, 1)),
        "raw_atlas_coverage_fraction": raw_coverage_fraction,
        "in_bounds_fraction": float(in_bounds_total / max(region_map.size, 1)),
        "pixel_area_um2": registration_slice.section_pixel_area_um2(atlas.voxel_size_um),
        "midline_quicknii_x": float(atlas.midline_quicknii_x),
        "marker_warp_enabled": marker_warp is not None,
        "marker_count_used": int(len(registration_slice.markers)),
        "region_smoothing_enabled": smooth_regions,
        "region_smoothing_kernel_size": int(smoothing_kernel_size),
        "region_smoothing_iterations": int(smoothing_iterations),
        "region_smoothing_downsample_factor": int(smoothing_downsample_factor),
        "region_contour_simplification_enabled": simplify_contours,
        "region_contour_simplification_tolerance_px": float(contour_tolerance_px),
        "region_contour_min_component_area_px": int(contour_min_component_area_px),
        "atlas_sampling_mode": atlas_sampling_mode,
        "atlas_sampling_radius_vox": int(atlas_sampling_radius_vox),
        "atlas_sampling_batch_size": int(atlas_sampling_batch_size),
    }
    return region_map, hemisphere_map, metrics


def _boundary_mask(region_map: np.ndarray) -> np.ndarray:
    boundaries = np.zeros_like(region_map, dtype=bool)
    boundaries[1:, :] |= region_map[1:, :] != region_map[:-1, :]
    boundaries[:, 1:] |= region_map[:, 1:] != region_map[:, :-1]
    return boundaries & (region_map > 0)


def _draw_objects(
    base: Image.Image,
    objects: list[DetectedObject],
    scale: float,
    draw_masks: bool,
    draw_centroids: bool,
    centroid_outline_rgb: tuple[int, int, int] = CELL_OUTLINE_RGB,
    mask_edge_rgb: tuple[int, int, int] = CELL_OUTLINE_RGB,
    centroid_mode: str = "circle",
) -> Image.Image:
    rendered = base.copy()
    draw = ImageDraw.Draw(rendered)
    pixels = rendered.load()
    for obj in objects:
        if draw_centroids:
            cx = int(round(obj.centroid_x_px * scale))
            cy = int(round(obj.centroid_y_px * scale))
            if centroid_mode == "point":
                radius = max(1, int(round(1.25 * scale)))
                draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    fill=centroid_outline_rgb,
                )
            else:
                radius = max(2, int(round(4 * scale)))
                draw.ellipse(
                    (cx - radius, cy - radius, cx + radius, cy + radius),
                    outline=centroid_outline_rgb,
                    width=max(1, int(round(scale))),
                )
        if draw_masks:
            mask = obj.mask_crop
            if scale != 1.0:
                resized = Image.fromarray((mask.astype(np.uint8) * 255)).resize(
                    (
                        max(1, int(round(mask.shape[1] * scale))),
                        max(1, int(round(mask.shape[0] * scale))),
                    ),
                    resample=Image.Resampling.NEAREST,
                )
                mask = np.asarray(resized) > 0
            edge = mask & ~ndimage.binary_erosion(mask)
            y0 = int(round(obj.bbox_origin[1] * scale))
            x0 = int(round(obj.bbox_origin[0] * scale))
            ys, xs = np.where(edge)
            for dy, dx in zip(ys, xs, strict=False):
                px = x0 + int(dx)
                py = y0 + int(dy)
                if 0 <= px < rendered.width and 0 <= py < rendered.height:
                    pixels[px, py] = mask_edge_rgb
    return rendered


def save_overlay_images(
    image: np.ndarray,
    region_map: np.ndarray,
    atlas: AtlasRepository,
    detected_objects: list[DetectedObject],
    preview_out: Path | None,
    full_out: Path,
    preview_max_size: int,
    full_max_size: int,
    include_region_fill_in_preview: bool,
    draw_masks: bool,
    draw_centroids: bool,
    png_compress_level: int = 9,
) -> tuple[Path | None, Path]:
    """Save preview and full-resolution QC overlay images."""

    full_out.parent.mkdir(parents=True, exist_ok=True)

    rgb = ensure_rgb(image)
    full_shape, full_scale = _scaled_output_shape(rgb.shape[:2], full_max_size)
    full_rgb = rgb
    full_region = region_map
    if full_shape != rgb.shape[:2]:
        full_rgb = np.asarray(
            Image.fromarray(rgb).resize((full_shape[1], full_shape[0]), resample=Image.Resampling.BILINEAR)
        )
        full_region = _resize_numeric_plane(region_map.astype(np.uint32, copy=False), full_shape, Image.Resampling.NEAREST)
    boundary_alpha = _antialiased_outline_alpha(full_region)

    full = _apply_outline_alpha(full_rgb.copy(), boundary_alpha, REGION_OUTLINE_RGB, alpha=0.9)
    full_image = Image.fromarray(full)
    full_image = _draw_objects(
        full_image,
        detected_objects,
        full_scale,
        draw_masks,
        draw_centroids,
        centroid_outline_rgb=CELL_OUTLINE_RGB,
        mask_edge_rgb=CELL_OUTLINE_RGB,
        centroid_mode="point",
    )
    _save_png(full_image, full_out, compress_level=png_compress_level)

    if preview_out is not None:
        preview_out.parent.mkdir(parents=True, exist_ok=True)
        scale = min(preview_max_size / max(rgb.shape[1], 1), preview_max_size / max(rgb.shape[0], 1), 1.0)
        preview_rgb = rgb
        preview_region = region_map
        if scale < 1.0:
            preview_size = (max(1, int(round(rgb.shape[1] * scale))), max(1, int(round(rgb.shape[0] * scale))))
            preview_rgb = np.asarray(Image.fromarray(rgb).resize(preview_size, resample=Image.Resampling.BILINEAR))
            preview_region = np.asarray(
                Image.fromarray(region_map.astype(np.uint32), mode="I").resize(preview_size, resample=Image.Resampling.NEAREST)
            )
        preview_boundary_alpha = _antialiased_outline_alpha(preview_region)
        preview_rgb = _apply_outline_alpha(preview_rgb, preview_boundary_alpha, REGION_OUTLINE_RGB, alpha=0.9)
        preview_image = Image.fromarray(preview_rgb)
        preview_image = _draw_objects(
            preview_image,
            detected_objects,
            scale,
            draw_masks,
            draw_centroids,
            centroid_outline_rgb=CELL_OUTLINE_RGB,
            mask_edge_rgb=CELL_OUTLINE_RGB,
            centroid_mode="point",
        )
        _save_png(preview_image, preview_out, compress_level=png_compress_level)

    return preview_out, full_out


def save_multichannel_overlay_images(
    channel_images: list[tuple[str, np.ndarray]],
    section_results: list,
    atlas: AtlasRepository,
    full_out: Path,
    full_max_size: int,
    draw_masks: bool,
    draw_centroids: bool,
    channel_colors: dict[str, list[int]] | None = None,
    png_compress_level: int = 9,
    tiff_compression: str = "tiff_adobe_deflate",
) -> dict[str, object]:
    """Save a consolidated multichannel atlas stack TIFF."""

    full_out.parent.mkdir(parents=True, exist_ok=True)
    if not channel_images or not section_results:
        raise ValueError("Multichannel overlay requires at least one image and one section result.")

    channel_colors = channel_colors or {}
    region_map = section_results[0].region_map
    stack_planes, channel_map = _build_multichannel_overlay_stack(
        channel_images=channel_images,
        section_results=section_results,
        region_map=region_map,
        atlas=atlas,
    )
    full_shape, _ = _scaled_output_shape(region_map.shape, full_max_size)
    full_stack_planes = stack_planes
    if full_shape != region_map.shape:
        continuous_indices = {
            int(channel_index) - 1
            for channel_index in channel_map.loc[channel_map["content"].astype(str).str.endswith("_raw_image"), "channel_index"]
        }
        full_stack_planes = _resize_stack_planes(stack_planes, full_shape, continuous_indices=continuous_indices)
    _save_stack_tiff(full_stack_planes, full_out, compression=tiff_compression)

    assets = {
        "full": full_out,
        "channel_map_frame": channel_map,
    }
    return assets


def save_multichannel_registered_label_images(
    section_results: list,
    atlas: AtlasRepository,
    numbered_preview_out: Path | None,
    numbered_full_out: Path,
    legend_out: Path | None,
    preview_max_size: int,
    channel_colors: dict[str, list[int]] | None = None,
    png_compress_level: int = 9,
    tiff_compression: str = "tiff_adobe_deflate",
) -> dict[str, object]:
    """Save a section-level atlas-numbered stack with per-channel ROI outlines."""

    numbered_full_out.parent.mkdir(parents=True, exist_ok=True)
    if numbered_preview_out is not None:
        numbered_preview_out.parent.mkdir(parents=True, exist_ok=True)
    if legend_out is not None:
        legend_out.parent.mkdir(parents=True, exist_ok=True)
    if not section_results:
        raise ValueError("Atlas-numbered multichannel export requires at least one section result.")

    channel_colors = channel_colors or {}
    region_map = section_results[0].region_map
    hemisphere_map = section_results[0].hemisphere_map
    label_map, legend_frame = _build_registered_label_map(region_map, hemisphere_map, atlas)

    total_channels = max(5, max(_channel_stack_index(result.bundle.image_channel or result.bundle.channel, index + 1) for index, result in enumerate(section_results)))
    stack_planes = [np.zeros(label_map.shape, dtype=np.int32) for _ in range(total_channels)]
    channel_rows: list[dict[str, object]] = []
    occupied: set[int] = set()
    for fallback_index, result in enumerate(section_results, start=1):
        channel_name = result.bundle.image_channel or result.bundle.channel
        plane_index = _channel_stack_index(channel_name, fallback_index)
        occupied.add(plane_index)
        stack_planes[plane_index - 1] = np.maximum(
            stack_planes[plane_index - 1],
            _objects_outline_plane(result.detected_objects, label_map.shape, plane_value=1),
        )
        channel_rows.append(
            {
                "channel_index": plane_index,
                "channel_name": f"CH{plane_index}",
                "content": f"{channel_name}_cell_roi_outline",
            }
        )

    atlas_plane_index = max(5, max(occupied, default=0) + 1)
    if atlas_plane_index > len(stack_planes):
        stack_planes.extend(np.zeros(label_map.shape, dtype=np.int32) for _ in range(atlas_plane_index - len(stack_planes)))
    stack_planes[atlas_plane_index - 1] = label_map.astype(np.int32, copy=False)
    channel_rows.append(
        {
            "channel_index": atlas_plane_index,
            "channel_name": f"CH{atlas_plane_index}",
            "content": "atlas_display_code",
        }
    )

    _save_stack_tiff(stack_planes, numbered_full_out, compression=tiff_compression)
    channel_map_frame = pd.DataFrame(channel_rows).sort_values("channel_index")

    if numbered_preview_out is not None:
        preview_image = _render_label_preview(label_map).convert("RGB")
        for index, result in enumerate(section_results):
            channel_name = result.bundle.image_channel or result.bundle.channel
            color = _resolve_channel_color(channel_name, channel_colors, index)
            preview_image = _draw_objects(
                preview_image,
                result.detected_objects,
                1.0,
                draw_masks=True,
                draw_centroids=False,
                centroid_outline_rgb=color,
                mask_edge_rgb=color,
                centroid_mode="point",
            )
        preview_scale = min(
            preview_max_size / max(label_map.shape[1], 1),
            preview_max_size / max(label_map.shape[0], 1),
            1.0,
        )
        if preview_scale < 1.0:
            preview_size = (
                max(1, int(round(label_map.shape[1] * preview_scale))),
                max(1, int(round(label_map.shape[0] * preview_scale))),
            )
            preview_image = preview_image.resize(preview_size, resample=Image.Resampling.BILINEAR)
        _save_png(preview_image, numbered_preview_out, compress_level=png_compress_level)

    if legend_out is not None:
        legend_frame.to_csv(legend_out, index=False)
    return {
        "numbered_preview": numbered_preview_out,
        "numbered_full": numbered_full_out,
        "legend_csv": legend_out,
        "channel_map_frame": channel_map_frame,
    }


def save_registered_label_images(
    region_map: np.ndarray,
    hemisphere_map: np.ndarray,
    atlas: AtlasRepository,
    numbered_preview_out: Path | None,
    numbered_full_out: Path,
    legend_out: Path | None,
    preview_max_size: int,
    png_compress_level: int = 9,
    tiff_compression: str = "tiff_adobe_deflate",
) -> dict[str, Path]:
    """Save a native registered atlas label image and section-specific legend."""

    numbered_full_out.parent.mkdir(parents=True, exist_ok=True)
    if numbered_preview_out is not None:
        numbered_preview_out.parent.mkdir(parents=True, exist_ok=True)
    if legend_out is not None:
        legend_out.parent.mkdir(parents=True, exist_ok=True)

    label_map, legend_frame = _build_registered_label_map(region_map, hemisphere_map, atlas)
    _save_label_image(label_map, numbered_full_out, compression=tiff_compression)

    if numbered_preview_out is not None:
        scale = min(
            preview_max_size / max(label_map.shape[1], 1),
            preview_max_size / max(label_map.shape[0], 1),
            1.0,
        )
        if scale < 1.0:
            preview_size = (
                max(1, int(round(label_map.shape[1] * scale))),
                max(1, int(round(label_map.shape[0] * scale))),
            )
            preview_label_map = np.asarray(
                Image.fromarray(label_map.astype(np.int32), mode="I").resize(preview_size, resample=Image.Resampling.NEAREST),
                dtype=np.int32,
            )
        else:
            preview_label_map = label_map

        _save_png(_render_label_preview(preview_label_map), numbered_preview_out, compress_level=png_compress_level)
    if legend_out is not None:
        legend_frame.to_csv(legend_out, index=False)
    return {
        "numbered_preview": numbered_preview_out,
        "numbered_full": numbered_full_out,
        "legend_csv": legend_out,
    }


def save_reference_overlay_images(
    image: np.ndarray,
    reference_atlas_image: np.ndarray,
    detected_objects: list[DetectedObject],
    reference_objects: list[DetectedObject],
    atlas: AtlasRepository | None,
    native_region_map: np.ndarray | None,
    native_hemisphere_map: np.ndarray | None,
    preview_out: Path | None,
    full_out: Path,
    numbered_preview_out: Path | None,
    numbered_full_out: Path,
    legend_out: Path | None,
    preview_max_size: int,
    full_max_size: int,
    overlay_alpha: float,
    mode_filter_size: int,
    roi_number_min_area_px: int,
    draw_masks: bool,
    draw_centroids: bool,
    png_compress_level: int = 9,
    tiff_compression: str = "tiff_adobe_deflate",
) -> dict[str, Path]:
    """Save overlay QC plus a raw atlas label image using an existing Nutil atlas map."""

    full_out.parent.mkdir(parents=True, exist_ok=True)
    numbered_full_out.parent.mkdir(parents=True, exist_ok=True)
    if preview_out is not None:
        preview_out.parent.mkdir(parents=True, exist_ok=True)
    if numbered_preview_out is not None:
        numbered_preview_out.parent.mkdir(parents=True, exist_ok=True)
    if legend_out is not None:
        legend_out.parent.mkdir(parents=True, exist_ok=True)

    rgb = ensure_rgb(image)
    cleaned_atlas = _clean_reference_atlas_image(reference_atlas_image, mode_filter_size=mode_filter_size)
    background_color = _dominant_background_color(cleaned_atlas)
    tissue_mask = np.any(cleaned_atlas != np.asarray(background_color, dtype=np.uint8), axis=2)
    patch_map, patch_infos = _build_reference_patch_map(
        cleaned_atlas,
        background_color=background_color,
        min_component_area_px=roi_number_min_area_px,
    )
    patch_infos = _attach_patch_reference_metadata(
        patch_map,
        patch_infos,
        reference_objects,
        atlas=atlas,
        native_region_map=native_region_map,
        native_hemisphere_map=native_hemisphere_map,
    )
    if legend_out is not None:
        _write_patch_legend(patch_infos, legend_out)

    boundary = _patch_boundary_mask(patch_map, tissue_mask)
    full_shape, full_scale = _scaled_output_shape(rgb.shape[:2], full_max_size)
    overlay_rgb = rgb.copy()
    overlay_boundary = boundary
    if full_shape != rgb.shape[:2]:
        overlay_rgb = np.asarray(
            Image.fromarray(overlay_rgb).resize((full_shape[1], full_shape[0]), resample=Image.Resampling.BILINEAR)
        )
        overlay_boundary = _resize_numeric_plane(boundary.astype(np.uint8, copy=False), full_shape, Image.Resampling.NEAREST) > 0
    overlay_rgb = _apply_outline_alpha(overlay_rgb, _antialias_mask(overlay_boundary), REGION_OUTLINE_RGB, alpha=0.9)
    overlay_image = _draw_objects(
        Image.fromarray(overlay_rgb),
        detected_objects,
        full_scale,
        draw_masks=False,
        draw_centroids=True,
        centroid_outline_rgb=CELL_OUTLINE_RGB,
        mask_edge_rgb=CELL_OUTLINE_RGB,
        centroid_mode="point",
    )
    _save_png(overlay_image, full_out, compress_level=png_compress_level)

    label_map = _build_patch_label_map(patch_map, patch_infos)
    _save_label_image(label_map, numbered_full_out, compression=tiff_compression)

    if preview_out is not None or numbered_preview_out is not None:
        scale = min(preview_max_size / max(rgb.shape[1], 1), preview_max_size / max(rgb.shape[0], 1), 1.0)
        if scale < 1.0:
            preview_size = (max(1, int(round(rgb.shape[1] * scale))), max(1, int(round(rgb.shape[0] * scale))))
            overlay_preview = overlay_image.resize(preview_size, resample=Image.Resampling.BILINEAR)
            preview_label_map = np.asarray(
                Image.fromarray(label_map.astype(np.int32), mode="I").resize(preview_size, resample=Image.Resampling.NEAREST),
                dtype=np.int32,
            )
        else:
            overlay_preview = overlay_image.copy()
            preview_label_map = label_map

        if preview_out is not None:
            _save_png(overlay_preview, preview_out, compress_level=png_compress_level)
        if numbered_preview_out is not None:
            _save_png(_render_label_preview(preview_label_map), numbered_preview_out, compress_level=png_compress_level)
    return {
        "overlay_preview": preview_out,
        "overlay_full": full_out,
        "numbered_preview": numbered_preview_out,
        "numbered_full": numbered_full_out,
        "legend_csv": legend_out,
    }


def _clean_reference_atlas_image(reference_atlas_image: np.ndarray, mode_filter_size: int) -> np.ndarray:
    rgb = ensure_rgb(reference_atlas_image)
    pil = Image.fromarray(rgb)
    if mode_filter_size >= 3:
        pil = pil.filter(ImageFilter.ModeFilter(size=mode_filter_size))
        pil = pil.filter(ImageFilter.ModeFilter(size=mode_filter_size))
    return np.asarray(pil)


def _dominant_background_color(rgb: np.ndarray) -> tuple[int, int, int]:
    flat = rgb.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    color = colors[int(np.argmax(counts))]
    return int(color[0]), int(color[1]), int(color[2])


def _build_reference_patch_map(
    atlas_rgb: np.ndarray,
    background_color: tuple[int, int, int],
    min_component_area_px: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    patch_map = np.zeros(atlas_rgb.shape[:2], dtype=np.uint16)
    patch_infos: list[dict[str, object]] = []
    next_patch_id = 1

    flat = atlas_rgb.reshape(-1, 3)
    unique_colors = np.unique(flat, axis=0)
    for color in unique_colors:
        color_tuple = tuple(int(value) for value in color)
        if color_tuple == background_color:
            continue
        color_mask = np.all(atlas_rgb == color, axis=2)
        labeled, n_labels = ndimage.label(color_mask)
        for label_index in range(1, n_labels + 1):
            component = labeled == label_index
            area_px = int(component.sum())
            if area_px < min_component_area_px:
                continue
            patch_map[component] = next_patch_id
            distance = ndimage.distance_transform_edt(component)
            label_y, label_x = np.unravel_index(int(np.argmax(distance)), distance.shape)
            patch_infos.append(
                {
                    "patch_id": next_patch_id,
                    "area_px": area_px,
                    "color_rgb": f"{color_tuple[0]},{color_tuple[1]},{color_tuple[2]}",
                    "label_x_px": int(label_x),
                    "label_y_px": int(label_y),
                }
            )
            next_patch_id += 1
    return patch_map, patch_infos


def _attach_patch_reference_metadata(
    patch_map: np.ndarray,
    patch_infos: list[dict[str, object]],
    reference_objects: list[DetectedObject],
    atlas: AtlasRepository | None = None,
    native_region_map: np.ndarray | None = None,
    native_hemisphere_map: np.ndarray | None = None,
) -> list[dict[str, object]]:
    region_ids = [int(info.get("dominant_region_id", 0)) for info in patch_infos]
    if atlas is not None and getattr(atlas, "regions", None):
        max_region_id = max(max(atlas.regions), max(region_ids, default=0))
    else:
        max_region_id = max(region_ids, default=0)
    midline_code_offset = midline_offset(max_region_id)

    patch_lookup = {int(item["patch_id"]): item for item in patch_infos}
    grouped: dict[int, list[DetectedObject]] = {}
    for obj in reference_objects:
        x = int(round(obj.centroid_x_px))
        y = int(round(obj.centroid_y_px))
        if not (0 <= x < patch_map.shape[1] and 0 <= y < patch_map.shape[0]):
            continue
        patch_id = int(patch_map[y, x])
        if patch_id == 0:
            continue
        grouped.setdefault(patch_id, []).append(obj)

    for patch_id, objects in grouped.items():
        info = patch_lookup[patch_id]
        counts = pd.Series(
            [f"{obj.region_id}|{obj.region_name}|{obj.hemisphere}" for obj in objects],
            dtype="string",
        ).value_counts()
        dominant_key = counts.index[0] if not counts.empty else "0|Unknown|unknown"
        dominant_region_id, dominant_region_name, dominant_hemisphere = dominant_key.split("|", 2)
        info["dominant_region_id"] = int(dominant_region_id)
        info["dominant_region_name"] = dominant_region_name
        info["dominant_hemisphere"] = dominant_hemisphere
        info["n_reference_objects"] = len(objects)
        info["assignment_source"] = "reference_objects"
        info["top_region_labels"] = "; ".join(
            f"{_format_region_pair_label(label)} ({int(count)})" for label, count in counts.head(3).items()
        )

    for info in patch_infos:
        if "dominant_region_id" not in info and native_region_map is not None and native_hemisphere_map is not None:
            patch_id = int(info["patch_id"])
            region_values = native_region_map[patch_map == patch_id]
            hemisphere_values = native_hemisphere_map[patch_map == patch_id]
            valid = region_values > 0
            if np.any(valid):
                pair_frame = pd.DataFrame(
                    {
                        "region_id": region_values[valid].astype(int),
                        "hemi_code": hemisphere_values[valid].astype(int),
                    }
                )
                pair_counts = pair_frame.value_counts().reset_index(name="count")
                top = pair_counts.iloc[0]
                hemisphere = {-1: "left", 0: "midline", 1: "right"}.get(int(top["hemi_code"]), "unknown")
                region = atlas.region_for_id(int(top["region_id"])) if atlas is not None else None
                info["dominant_region_id"] = int(top["region_id"])
                info["dominant_region_name"] = region.name if region is not None else "Unknown"
                info["dominant_hemisphere"] = hemisphere
                info["assignment_source"] = "native_region_map"
        info.setdefault("dominant_region_id", 0)
        info.setdefault("dominant_region_name", "Unknown")
        info.setdefault(
            "dominant_hemisphere",
            _infer_patch_hemisphere(float(info["label_x_px"]), patch_map.shape[1]),
        )
        info.setdefault("assignment_source", "position_fallback")
        info.setdefault("n_reference_objects", 0)
        info.setdefault("top_region_labels", "")
        if not info["dominant_region_name"]:
            info["dominant_region_name"] = "Unknown"
        info["display_code"] = stable_region_display_code(
            int(info["dominant_region_id"]),
            str(info["dominant_hemisphere"]),
            midline_code_offset,
        )
        info["midline_code_offset"] = midline_code_offset
    return patch_infos


def _write_patch_legend(patch_infos: list[dict[str, object]], legend_out: Path) -> None:
    frame = pd.DataFrame(patch_infos)
    if not frame.empty and "patch_id" in frame.columns:
        frame = frame.sort_values("patch_id")
    frame.to_csv(legend_out, index=False)


def _patch_boundary_mask(patch_map: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    boundaries = np.zeros_like(patch_map, dtype=bool)
    boundaries[1:, :] |= patch_map[1:, :] != patch_map[:-1, :]
    boundaries[:, 1:] |= patch_map[:, 1:] != patch_map[:, :-1]
    boundaries &= tissue_mask
    return ndimage.binary_dilation(boundaries, iterations=1)


def _blend_reference_atlas(
    image_rgb: np.ndarray,
    atlas_rgb: np.ndarray,
    tissue_mask: np.ndarray,
    overlay_alpha: float,
) -> np.ndarray:
    blended = image_rgb.astype(np.float32).copy()
    atlas_float = atlas_rgb.astype(np.float32, copy=False)
    blended[tissue_mask] = blended[tissue_mask] * (1.0 - overlay_alpha) + atlas_float[tissue_mask] * overlay_alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _build_patch_label_map(patch_map: np.ndarray, patch_infos: list[dict[str, object]]) -> np.ndarray:
    label_map = np.zeros(patch_map.shape, dtype=np.int32)
    for info in patch_infos:
        patch_id = int(info["patch_id"])
        label_map[patch_map == patch_id] = int(info.get("display_code", 0))
    return label_map


def _build_registered_label_map(
    region_map: np.ndarray,
    hemisphere_map: np.ndarray,
    atlas: AtlasRepository,
) -> tuple[np.ndarray, pd.DataFrame]:
    label_map = np.zeros(region_map.shape, dtype=np.int32)
    max_region_id = max(atlas.regions) if getattr(atlas, "regions", None) else int(region_map.max(initial=0))
    midline_code_offset = midline_offset(max_region_id)
    legend_rows: list[dict[str, object]] = []

    for hemisphere_code, hemisphere_name in ((-1, "left"), (1, "right")):
        hemi_mask = region_map > 0
        hemi_mask &= hemisphere_map == hemisphere_code
        if not np.any(hemi_mask):
            continue
        region_ids, pixel_counts = np.unique(region_map[hemi_mask], return_counts=True)
        for region_id, pixel_count in zip(region_ids, pixel_counts, strict=False):
            region_id_int = int(region_id)
            display_code = stable_region_display_code(region_id_int, hemisphere_name, midline_code_offset)
            pair_mask = hemi_mask & (region_map == region_id_int)
            label_map[pair_mask] = display_code
            region = atlas.region_for_id(region_id_int)
            legend_rows.append(
                {
                    "display_code": display_code,
                    "region_id": region_id_int,
                    "region_name": region.name if region is not None else "Unknown",
                    "hemisphere": hemisphere_name,
                    "pixel_count": int(pixel_count),
                    "parent_region_id": region.parent_id if region is not None else None,
                    "hierarchy": " > ".join(region.hierarchy_names) if region is not None else "",
                    "midline_code_offset": midline_code_offset,
                    "encoding_rule": "right=+region_id,left=-region_id",
                    "source": "native_registered_map",
                }
            )

    legend_frame = pd.DataFrame(legend_rows)
    if not legend_frame.empty:
        legend_frame = legend_frame.sort_values(["hemisphere", "region_id"]).reset_index(drop=True)
    return label_map, legend_frame


def _scaled_output_shape(shape_hw: tuple[int, int], max_size: int) -> tuple[tuple[int, int], float]:
    height, width = int(shape_hw[0]), int(shape_hw[1])
    if max_size <= 0:
        return (height, width), 1.0
    scale = min(max_size / max(width, 1), max_size / max(height, 1), 1.0)
    if scale >= 1.0:
        return (height, width), 1.0
    return (
        max(1, int(round(height * scale))),
        max(1, int(round(width * scale))),
    ), float(scale)


def _resize_numeric_plane(
    plane: np.ndarray,
    output_shape: tuple[int, int],
    resample: Image.Resampling,
) -> np.ndarray:
    if plane.shape == output_shape:
        return plane
    resized = Image.fromarray(np.asarray(plane, dtype=np.float32), mode="F").resize(
        (output_shape[1], output_shape[0]),
        resample=resample,
    )
    array = np.asarray(resized, dtype=np.float32)
    if np.issubdtype(plane.dtype, np.integer):
        return np.rint(array).astype(plane.dtype)
    return array.astype(plane.dtype)


def _resize_stack_planes(
    stack_planes: list[np.ndarray],
    output_shape: tuple[int, int],
    continuous_indices: set[int] | None = None,
) -> list[np.ndarray]:
    continuous_indices = continuous_indices or set()
    resized_planes: list[np.ndarray] = []
    for index, plane in enumerate(stack_planes):
        resample = Image.Resampling.BILINEAR if index in continuous_indices else Image.Resampling.NEAREST
        resized_planes.append(_resize_numeric_plane(np.asarray(plane), output_shape, resample))
    return resized_planes


def _save_png(image: Image.Image, out_path: Path, compress_level: int = 9) -> None:
    image.save(out_path, optimize=True, compress_level=max(0, min(int(compress_level), 9)))


def _save_label_image(label_map: np.ndarray, out_path: Path, compression: str | None = None) -> None:
    tifffile.imwrite(
        str(out_path),
        label_map.astype(np.int32, copy=False),
        compression=_normalize_tiff_compression(compression),
        metadata=None,
    )


def _render_label_preview(label_map: np.ndarray) -> Image.Image:
    preview = np.zeros(label_map.shape, dtype=np.uint8)
    nonzero_codes = sorted(int(code) for code in np.unique(label_map) if int(code) != 0)
    if nonzero_codes:
        ramp = np.linspace(24, 255, num=len(nonzero_codes), dtype=np.uint8)
        for value, code in zip(ramp, nonzero_codes, strict=False):
            preview[label_map == code] = value
    return Image.fromarray(preview, mode="L")


def _resolve_channel_color(
    channel_name: str,
    channel_colors: dict[str, list[int]],
    index: int,
) -> tuple[int, int, int]:
    normalized = {key.upper(): tuple(int(value) for value in values[:3]) for key, values in channel_colors.items()}
    if channel_name.upper() in normalized:
        return normalized[channel_name.upper()]
    return FALLBACK_CHANNEL_COLORS[index % len(FALLBACK_CHANNEL_COLORS)]


def _resize_grayscale_to_shape(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    if image.shape == output_shape:
        return image.astype(np.float32, copy=False)
    resized = Image.fromarray(image.astype(np.float32), mode="F").resize(
        (output_shape[1], output_shape[0]),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.float32)


def _normalize_intensity_image(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    intensity = grayscale_intensity(image).astype(np.float32, copy=False)
    intensity = _resize_grayscale_to_shape(intensity, output_shape)
    positive = intensity[intensity > 0]
    if positive.size == 0:
        return np.zeros(output_shape, dtype=np.float32)
    low = float(np.percentile(positive, 1.0))
    high = float(np.percentile(positive, 99.5))
    if high <= low:
        high = float(positive.max())
        low = float(positive.min())
    scale = max(high - low, 1e-6)
    return np.clip((intensity - low) / scale, 0.0, 1.0)


def _compose_multichannel_rgb(
    channel_images: list[tuple[str, np.ndarray]],
    channel_colors: dict[str, list[int]],
) -> np.ndarray:
    base_shape = channel_images[0][1].shape[:2]
    composite = np.zeros((base_shape[0], base_shape[1], 3), dtype=np.float32)
    for index, (channel_name, image) in enumerate(channel_images):
        normalized = _normalize_intensity_image(image, base_shape)
        color = np.asarray(_resolve_channel_color(channel_name, channel_colors, index), dtype=np.float32)
        composite += normalized[..., None] * color[None, None, :]
    return np.clip(composite, 0, 255).astype(np.uint8)


def _smooth_label_map(
    label_map: np.ndarray,
    kernel_size: int,
    iterations: int,
    chunk_rows: int,
    downsample_factor: int = 1,
) -> np.ndarray:
    """Smooth a registered label map before quantification and visualization."""

    if kernel_size < 3 or kernel_size % 2 == 0 or iterations < 1:
        return label_map

    smoothed = label_map.astype(np.uint32, copy=True)
    height, width = smoothed.shape

    if downsample_factor >= 2:
        smoothed = _coarse_label_resample(smoothed, factor=downsample_factor)

    pad = kernel_size // 2
    n_neighbors = kernel_size * kernel_size

    for _ in range(iterations):
        tissue_mask = smoothed > 0
        structure = ndimage.generate_binary_structure(2, 1)
        tissue_mask = ndimage.binary_closing(tissue_mask, structure=structure, iterations=max(1, pad))
        tissue_mask = ndimage.binary_opening(tissue_mask, structure=structure, iterations=max(1, pad // 2))
        tissue_mask = ndimage.binary_fill_holes(tissue_mask)
        padded = np.pad(smoothed, ((pad, pad), (pad, pad)), mode="edge")
        updated = smoothed.copy()

        for row_start in range(0, height, chunk_rows):
            row_end = min(row_start + chunk_rows, height)
            tissue_chunk = tissue_mask[row_start:row_end]
            if not np.any(tissue_chunk):
                continue

            padded_chunk = padded[row_start : row_end + (2 * pad), :]
            windows = np.lib.stride_tricks.sliding_window_view(padded_chunk, (kernel_size, kernel_size))
            neighborhoods = windows.reshape(row_end - row_start, width, n_neighbors)
            target_neighborhoods = neighborhoods[tissue_chunk]
            current_values = smoothed[row_start:row_end][tissue_chunk]
            sorted_neighbors = np.sort(target_neighborhoods, axis=1)
            if _best_replacement_values_numba is not None:
                best_values, best_counts, current_counts = _best_replacement_values_numba(
                    sorted_neighbors.astype(np.uint32, copy=False),
                    current_values.astype(np.uint32, copy=False),
                )
            else:
                run_lengths = np.ones(sorted_neighbors.shape, dtype=np.uint8)
                for index in range(1, n_neighbors):
                    same_as_previous = sorted_neighbors[:, index] == sorted_neighbors[:, index - 1]
                    run_lengths[:, index] = np.where(same_as_previous, run_lengths[:, index - 1] + 1, 1)
                run_lengths[sorted_neighbors == 0] = 0
                best_index = np.argmax(run_lengths, axis=1)
                best_values = sorted_neighbors[np.arange(len(best_index)), best_index]
                best_counts = run_lengths[np.arange(len(best_index)), best_index].astype(np.int16)
                current_counts = np.sum(target_neighborhoods == current_values[:, None], axis=1, dtype=np.int16)
            replacement_values = np.where(
                (best_values != 0) & (best_counts > current_counts),
                best_values,
                current_values,
            )

            updated_chunk = updated[row_start:row_end]
            updated_chunk[tissue_chunk] = replacement_values.astype(np.uint32, copy=False)
            updated[row_start:row_end] = updated_chunk
        updated[~tissue_mask] = 0
        smoothed = updated

    return smoothed


def _simplify_label_map_contours(
    label_map: np.ndarray,
    tolerance_px: float,
    min_component_area_px: int,
) -> np.ndarray:
    """Simplify region contours and re-rasterize them back into label space."""

    if tolerance_px <= 0 or min_component_area_px < 1:
        return label_map

    tissue_mask = label_map > 0
    if not np.any(tissue_mask):
        return label_map

    candidate = np.zeros_like(label_map, dtype=np.uint32)
    structure = ndimage.generate_binary_structure(2, 2)
    expansion = max(1, int(np.ceil(tolerance_px)))
    region_ids, counts = np.unique(label_map[tissue_mask], return_counts=True)
    region_items = sorted(
        ((int(region_id), int(count)) for region_id, count in zip(region_ids, counts, strict=False)),
        key=lambda item: item[1],
        reverse=True,
    )

    for region_id, region_area in region_items:
        if region_area < min_component_area_px:
            continue
        region_mask = label_map == region_id
        component_labels, _ = ndimage.label(region_mask, structure=structure)
        component_slices = ndimage.find_objects(component_labels)
        for component_index, component_slice in enumerate(component_slices, start=1):
            if component_slice is None:
                continue
            local_component = component_labels[component_slice] == component_index
            if int(local_component.sum()) < min_component_area_px:
                continue
            simplified_component = _simplify_component_mask(local_component, tolerance_px)
            allowed_band = ndimage.binary_dilation(local_component, structure=structure, iterations=expansion)
            claim_mask = simplified_component & allowed_band
            local_candidate = candidate[component_slice]
            local_candidate[claim_mask] = np.uint32(region_id)
            candidate[component_slice] = local_candidate

    boundary_band = ndimage.binary_dilation(
        _boundary_mask(label_map),
        structure=structure,
        iterations=max(1, expansion + 1),
    )
    blended = label_map.copy()
    update_mask = boundary_band & (candidate > 0)
    blended[update_mask] = candidate[update_mask]
    blended[~tissue_mask] = 0
    return blended


def _simplify_component_mask(component_mask: np.ndarray, tolerance_px: float) -> np.ndarray:
    """Simplify one connected component by polygonizing its contour and rasterizing it again."""

    if component_mask.size == 0 or not np.any(component_mask):
        return component_mask

    filled_component = ndimage.binary_fill_holes(component_mask)
    outer_loops = _trace_boundary_loops(filled_component)
    if not outer_loops:
        return component_mask

    canvas = Image.new("L", (component_mask.shape[1] + 1, component_mask.shape[0] + 1), 0)
    draw = ImageDraw.Draw(canvas)
    for loop in outer_loops:
        simplified_loop = _simplify_polygon(loop, tolerance_px)
        if len(simplified_loop) >= 3:
            draw.polygon([tuple(point) for point in simplified_loop], fill=1)

    hole_mask = filled_component & ~component_mask
    if np.any(hole_mask):
        hole_labels, _ = ndimage.label(hole_mask, structure=ndimage.generate_binary_structure(2, 2))
        for hole_index, hole_slice in enumerate(ndimage.find_objects(hole_labels), start=1):
            if hole_slice is None:
                continue
            local_hole = hole_labels[hole_slice] == hole_index
            hole_loops = _trace_boundary_loops(local_hole)
            for loop in hole_loops:
                offset_loop = loop + np.array([hole_slice[1].start, hole_slice[0].start], dtype=np.float64)
                simplified_loop = _simplify_polygon(offset_loop, tolerance_px)
                if len(simplified_loop) >= 3:
                    draw.polygon([tuple(point) for point in simplified_loop], fill=0)

    simplified = np.asarray(canvas, dtype=np.uint8)[: component_mask.shape[0], : component_mask.shape[1]] > 0
    union = np.logical_or(simplified, component_mask)
    if not np.any(union):
        return component_mask
    iou = float(np.logical_and(simplified, component_mask).sum() / union.sum())
    if iou < 0.85:
        return component_mask
    return simplified


def _trace_boundary_loops(mask: np.ndarray) -> list[np.ndarray]:
    """Trace oriented boundary loops around a binary mask."""

    if mask.size == 0 or not np.any(mask):
        return []

    ys, xs = np.where(mask)
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    height, width = mask.shape
    for y, x in zip(ys.tolist(), xs.tolist(), strict=False):
        if y == 0 or not mask[y - 1, x]:
            edges.append(((x, y), (x + 1, y)))
        if x == width - 1 or not mask[y, x + 1]:
            edges.append(((x + 1, y), (x + 1, y + 1)))
        if y == height - 1 or not mask[y + 1, x]:
            edges.append(((x + 1, y + 1), (x, y + 1)))
        if x == 0 or not mask[y, x - 1]:
            edges.append(((x, y + 1), (x, y)))

    if not edges:
        return []

    starts: dict[tuple[int, int], list[int]] = {}
    for edge_index, (start, _) in enumerate(edges):
        starts.setdefault(start, []).append(edge_index)

    used = np.zeros(len(edges), dtype=bool)
    loops: list[np.ndarray] = []
    for edge_index, (start, end) in enumerate(edges):
        if used[edge_index]:
            continue
        used[edge_index] = True
        loop: list[tuple[int, int]] = [start, end]
        previous_direction = (end[0] - start[0], end[1] - start[1])
        current = end
        while current != start:
            candidate_indices = [candidate for candidate in starts.get(current, []) if not used[candidate]]
            if not candidate_indices:
                break
            next_index = _select_next_edge(previous_direction, edges, candidate_indices)
            used[next_index] = True
            _, next_point = edges[next_index]
            loop.append(next_point)
            previous_direction = (next_point[0] - current[0], next_point[1] - current[1])
            current = next_point
        if current == start and len(loop) >= 4:
            loops.append(np.asarray(loop, dtype=np.float64))
    return loops


def _select_next_edge(
    previous_direction: tuple[int, int],
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    candidate_indices: list[int],
) -> int:
    """Choose the next oriented edge while following a clockwise boundary."""

    previous_index = _direction_index(previous_direction)
    ranked_candidates: list[tuple[int, int]] = []
    for candidate_index in candidate_indices:
        start, end = edges[candidate_index]
        candidate_direction = (end[0] - start[0], end[1] - start[1])
        turn = (_direction_index(candidate_direction) - previous_index) % 4
        ranked_candidates.append((turn, candidate_index))
    ranked_candidates.sort(key=lambda item: item[0])
    return ranked_candidates[0][1]


def _direction_index(direction: tuple[int, int]) -> int:
    mapping = {
        (1, 0): 0,
        (0, 1): 1,
        (-1, 0): 2,
        (0, -1): 3,
    }
    return mapping.get(direction, 0)


def _simplify_polygon(points: np.ndarray, tolerance_px: float) -> np.ndarray:
    """Prune vertices that stay within tolerance of neighboring segments."""

    ring = _remove_duplicate_vertices(points)
    ring = _remove_collinear_vertices(ring)
    if len(ring) <= 3:
        return ring

    while len(ring) > 3:
        distances = np.asarray(
            [
                _point_segment_distance(ring[index], ring[index - 1], ring[(index + 1) % len(ring)])
                for index in range(len(ring))
            ],
            dtype=np.float64,
        )
        remove_index = int(np.argmin(distances))
        if float(distances[remove_index]) > tolerance_px:
            break
        ring = np.delete(ring, remove_index, axis=0)
        ring = _remove_collinear_vertices(ring)
        if len(ring) <= 3:
            break
    return ring


def _remove_duplicate_vertices(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    filtered = [points[0]]
    for point in points[1:]:
        if not np.allclose(point, filtered[-1]):
            filtered.append(point)
    result = np.asarray(filtered, dtype=np.float64)
    if len(result) > 1 and np.allclose(result[0], result[-1]):
        result = result[:-1]
    return result


def _remove_collinear_vertices(points: np.ndarray) -> np.ndarray:
    if len(points) <= 3:
        return points
    keep = np.ones(len(points), dtype=bool)
    for index in range(len(points)):
        previous_point = points[index - 1]
        current_point = points[index]
        next_point = points[(index + 1) % len(points)]
        previous_vector = current_point - previous_point
        next_vector = next_point - current_point
        cross = (previous_vector[0] * next_vector[1]) - (previous_vector[1] * next_vector[0])
        if abs(float(cross)) <= 1e-6 and float(np.dot(previous_vector, next_vector)) >= 0:
            keep[index] = False
    if int(keep.sum()) < 3:
        return points
    return points[keep]


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    segment_norm = float(np.dot(segment, segment))
    if segment_norm <= 1e-12:
        return float(np.linalg.norm(point - start))
    projection = float(np.dot(point - start, segment) / segment_norm)
    projection = min(1.0, max(0.0, projection))
    closest = start + (projection * segment)
    return float(np.linalg.norm(point - closest))


def _apply_outline(
    image_rgb: np.ndarray,
    boundary_mask: np.ndarray,
    outline_rgb: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    outlined = image_rgb.astype(np.float32, copy=True)
    outline = np.asarray(outline_rgb, dtype=np.float32)
    outlined[boundary_mask] = outlined[boundary_mask] * (1.0 - alpha) + outline * alpha
    return np.clip(outlined, 0, 255).astype(np.uint8)


def _apply_outline_alpha(
    image_rgb: np.ndarray,
    alpha_mask: np.ndarray,
    outline_rgb: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    outlined = image_rgb.astype(np.float32, copy=True)
    outline = np.asarray(outline_rgb, dtype=np.float32)
    effective_alpha = np.clip(alpha_mask.astype(np.float32, copy=False) * float(alpha), 0.0, 1.0)
    outlined = outlined * (1.0 - effective_alpha[..., None]) + outline[None, None, :] * effective_alpha[..., None]
    return np.clip(outlined, 0, 255).astype(np.uint8)


def _antialias_mask(mask: np.ndarray, scale_factor: int = 4, dilation_width_px: int = 1) -> np.ndarray:
    if mask.size == 0:
        return np.zeros(mask.shape, dtype=np.float32)
    base = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    if dilation_width_px > 0:
        filter_size = max(3, (dilation_width_px * 2) + 1)
        base = base.filter(ImageFilter.MaxFilter(size=filter_size))
    high = base.resize(
        (max(1, mask.shape[1] * scale_factor), max(1, mask.shape[0] * scale_factor)),
        resample=Image.Resampling.NEAREST,
    )
    high = high.filter(ImageFilter.MaxFilter(size=max(3, scale_factor + 1 if (scale_factor + 1) % 2 else scale_factor + 2)))
    low = high.resize((mask.shape[1], mask.shape[0]), resample=Image.Resampling.LANCZOS)
    return np.asarray(low, dtype=np.float32) / 255.0


def _antialiased_outline_alpha(region_map: np.ndarray) -> np.ndarray:
    return _antialias_mask(_boundary_mask(region_map), scale_factor=4, dilation_width_px=0)


def _coarse_label_resample(label_map: np.ndarray, factor: int) -> np.ndarray:
    if factor < 2:
        return label_map

    height, width = label_map.shape
    pad_h = (-height) % factor
    pad_w = (-width) % factor
    padded = np.pad(label_map, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
    out_h = padded.shape[0] // factor
    out_w = padded.shape[1] // factor
    blocks = padded.reshape(out_h, factor, out_w, factor).transpose(0, 2, 1, 3).reshape(out_h, out_w, factor * factor)
    coarse = _majority_labels(blocks.reshape(-1, factor * factor)).reshape(out_h, out_w)
    expanded = np.repeat(np.repeat(coarse, factor, axis=0), factor, axis=1)[:height, :width]
    if expanded.shape != label_map.shape:
        expanded = expanded[:height, :width]
    return expanded.astype(np.uint32, copy=False)


def _majority_labels(values: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values, axis=1)
    run_lengths = np.ones(sorted_values.shape, dtype=np.uint16)
    for index in range(1, sorted_values.shape[1]):
        same_as_previous = sorted_values[:, index] == sorted_values[:, index - 1]
        run_lengths[:, index] = np.where(same_as_previous, run_lengths[:, index - 1] + 1, 1)
    run_lengths[sorted_values == 0] = 0
    best_index = np.argmax(run_lengths, axis=1)
    best_values = sorted_values[np.arange(sorted_values.shape[0]), best_index]
    return best_values.astype(np.uint32, copy=False)


def _channel_stack_index(channel_name: str, fallback_index: int) -> int:
    match = CHANNEL_PATTERN.search(channel_name or "")
    if match:
        return max(1, int(match.group(1)))
    return max(1, fallback_index)


def _stack_intensity_plane(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    normalized = _normalize_intensity_image(image, output_shape)
    return np.rint(normalized * 65535.0).astype(np.int32)


def _objects_point_plane(
    objects: list[DetectedObject],
    output_shape: tuple[int, int],
    point_value: int = 65535,
) -> np.ndarray:
    plane = np.zeros(output_shape, dtype=np.int32)
    for obj in objects:
        cx = int(round(obj.centroid_x_px))
        cy = int(round(obj.centroid_y_px))
        if 0 <= cx < output_shape[1] and 0 <= cy < output_shape[0]:
            plane[cy, cx] = point_value
    plane = ndimage.maximum_filter(plane, size=3)
    return plane


def _objects_outline_plane(
    objects: list[DetectedObject],
    output_shape: tuple[int, int],
    plane_value: int = 1,
) -> np.ndarray:
    plane = np.zeros(output_shape, dtype=np.int32)
    for obj in objects:
        edge = obj.mask_crop & ~ndimage.binary_erosion(obj.mask_crop)
        x0, y0 = obj.bbox_origin
        ys, xs = np.where(edge)
        y_coords = y0 + ys
        x_coords = x0 + xs
        valid = (
            (x_coords >= 0)
            & (x_coords < output_shape[1])
            & (y_coords >= 0)
            & (y_coords < output_shape[0])
        )
        plane[y_coords[valid], x_coords[valid]] = plane_value
    return plane


def _objects_fill_plane(
    objects: list[DetectedObject],
    output_shape: tuple[int, int],
    plane_value: int = 65535,
) -> np.ndarray:
    plane = np.zeros(output_shape, dtype=np.int32)
    for obj in objects:
        mask = np.asarray(obj.mask_crop, dtype=bool)
        if not np.any(mask):
            continue
        x0, y0 = obj.bbox_origin
        ys, xs = np.where(mask)
        y_coords = y0 + ys
        x_coords = x0 + xs
        valid = (
            (x_coords >= 0)
            & (x_coords < output_shape[1])
            & (y_coords >= 0)
            & (y_coords < output_shape[0])
        )
        plane[y_coords[valid], x_coords[valid]] = plane_value
    return plane


def _save_stack_tiff(stack_planes: list[np.ndarray], out_path: Path, compression: str | None = None) -> None:
    if not stack_planes:
        raise ValueError("Cannot save an empty TIFF stack.")
    stack = np.stack([np.asarray(plane, dtype=np.int32) for plane in stack_planes], axis=0)
    tifffile.imwrite(
        str(out_path),
        stack,
        compression=_normalize_tiff_compression(compression),
        metadata=None,
    )


def _build_multichannel_overlay_stack(
    channel_images: list[tuple[str, np.ndarray]],
    section_results: list,
    region_map: np.ndarray,
    atlas: AtlasRepository,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    hemisphere_map = section_results[0].hemisphere_map
    label_map, _ = _build_registered_label_map(region_map, hemisphere_map, atlas)
    stack_planes: list[np.ndarray] = []
    rows: list[dict[str, object]] = []

    def append_plane(
        plane: np.ndarray,
        *,
        content: str,
        notes: str,
        source_channel: str = "",
    ) -> None:
        plane_index = len(stack_planes) + 1
        stack_planes.append(np.asarray(plane, dtype=np.int32))
        rows.append(
            {
                "channel_index": plane_index,
                "channel_name": f"CH{plane_index}",
                "content": content,
                "source_channel": source_channel,
                "notes": notes,
            }
        )

    sorted_channel_images = sorted(
        enumerate(channel_images, start=1),
        key=lambda item: _channel_stack_index(item[1][0], item[0]),
    )
    sorted_results = sorted(
        section_results,
        key=lambda result: _channel_stack_index(result.bundle.image_channel or result.bundle.channel, 1),
    )

    for _, (channel_name, image) in sorted_channel_images:
        append_plane(
            _stack_intensity_plane(image, region_map.shape),
            content=f"{channel_name}_raw_image",
            source_channel=channel_name,
            notes="Original image intensity plane.",
        )

    for result in sorted_results:
        channel_name = result.bundle.image_channel or result.bundle.channel
        append_plane(
            _objects_fill_plane(result.detected_objects, region_map.shape, plane_value=65535),
            content=f"{channel_name}_cell_roi_outline",
            source_channel=channel_name,
            notes="Detected cell ROI mask for this channel.",
        )

    overlap_objects = [
        obj
        for result in section_results
        for obj in result.detected_objects
        if len({channel.upper() for channel in obj.matched_channels}) > 1
    ]
    if overlap_objects:
        overlap_groups = sorted({str(obj.overlap_group) for obj in overlap_objects if str(obj.overlap_group)})
        append_plane(
            _objects_fill_plane(overlap_objects, region_map.shape, plane_value=65535),
            content="matched_overlap_roi_outline",
            notes=", ".join(overlap_groups) if overlap_groups else "Matched multi-channel ROI masks.",
        )

    append_plane(
        np.rint(_antialiased_outline_alpha(region_map) * 65535.0).astype(np.int32),
        content="brain_region_outline_edge_roi",
        notes="Allen atlas border and edge ROI after registration.",
    )

    append_plane(
        label_map.astype(np.int32, copy=False),
        content="atlas_display_code",
        notes="Signed display codes. See atlas_display_codebook.csv for ID mapping.",
    )

    channel_map = pd.DataFrame(rows).drop_duplicates(subset=["channel_index", "content"]).sort_values("channel_index")
    return stack_planes, channel_map


def _save_multichannel_channel_views(
    channel_images: list[tuple[str, np.ndarray]],
    section_results: list,
    preview_out: Path,
    preview_max_size: int,
    channel_colors: dict[str, list[int]],
    draw_masks: bool,
    draw_centroids: bool,
) -> dict[str, Path]:
    channel_image_lookup = {channel_name.upper(): image for channel_name, image in channel_images}
    result_lookup = {
        (result.bundle.image_channel or result.bundle.channel).upper(): result for result in section_results
    }
    assets: dict[str, Path] = {}
    ordered_channels = [
        channel_name
        for channel_name, _ in sorted(
            channel_images,
            key=lambda item: _channel_stack_index(item[0], 1),
        )
    ]
    for subset in [[channel_name] for channel_name in ordered_channels] + [ordered_channels]:
        subset_key = "".join(channel.upper() for channel in subset)
        subset_images = [(channel, channel_image_lookup[channel.upper()]) for channel in subset]
        subset_results = [result_lookup[channel.upper()] for channel in subset if channel.upper() in result_lookup]
        if not subset_images or not subset_results:
            continue
        base_rgb = _compose_multichannel_rgb(subset_images, channel_colors)
        region_alpha = _antialiased_outline_alpha(subset_results[0].region_map)
        view_rgb = _apply_outline_alpha(base_rgb, region_alpha, REGION_OUTLINE_RGB, alpha=0.9)
        view_image = Image.fromarray(view_rgb)
        for index, result in enumerate(subset_results):
            channel_name = result.bundle.image_channel or result.bundle.channel
            color = _resolve_channel_color(channel_name, channel_colors, index)
            view_image = _draw_objects(
                view_image,
                result.detected_objects,
                1.0,
                draw_masks=draw_masks,
                draw_centroids=draw_centroids,
                centroid_outline_rgb=color,
                mask_edge_rgb=color,
                centroid_mode="point",
            )
        scale = min(
            preview_max_size / max(view_image.width, 1),
            preview_max_size / max(view_image.height, 1),
            1.0,
        )
        if scale < 1.0:
            view_image = view_image.resize(
                (
                    max(1, int(round(view_image.width * scale))),
                    max(1, int(round(view_image.height * scale))),
                ),
                resample=Image.Resampling.BILINEAR,
            )
        view_path = preview_out.with_name(f"{preview_out.stem}_{subset_key}.png")
        view_image.save(view_path)
        assets[f"view_{subset_key}"] = view_path
    return assets


def _infer_patch_hemisphere(label_x_px: float, width: int) -> str:
    return "right" if label_x_px >= (width / 2.0) else "left"


def _normalize_tiff_compression(compression: str | None) -> str | None:
    if not compression:
        return None
    normalized = str(compression).strip().lower()
    if normalized.startswith("tiff_"):
        normalized = normalized[5:]
    return normalized or None


def _format_region_pair_label(serialized: str) -> str:
    region_id, region_name, hemisphere = serialized.split("|", 2)
    return f"{region_name} [{hemisphere}] ({region_id})"
