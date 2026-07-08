"""Connected-component based object detection from binary masks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
try:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
except Exception:  # pragma: no cover - optional until watershed mode is used
    peak_local_max = None
    watershed = None

from data_models.models import DetectedObject


def _label_connected_mask(
    mask: np.ndarray,
    apply_watershed: bool = False,
    watershed_marker_threshold_px: float | str = "auto",
    min_component_area_px: int = 1,
    watershed_selective_area_percentile: float = 90.0,
    watershed_selective_elongation_threshold: float = 2.0,
) -> tuple[np.ndarray, int]:
    binary = np.asarray(mask, dtype=bool)
    if not apply_watershed:
        return ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))

    if watershed is None or peak_local_max is None:
        raise ImportError("Watershed mode requires scikit-image. Install scikit-image>=0.22.")

    labeled, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if count <= 0:
        return labeled, count

    component_meta = _component_metadata(labeled, min_component_area_px=min_component_area_px)
    if not component_meta:
        return labeled, count
    area_threshold = _resolve_selective_area_threshold(
        component_meta,
        min_component_area_px=min_component_area_px,
        watershed_selective_area_percentile=watershed_selective_area_percentile,
    )
    elongation_threshold = max(1.0, float(watershed_selective_elongation_threshold))

    out = np.zeros(binary.shape, dtype=np.int32)
    next_label = 1
    for meta in component_meta:
        bbox = meta["bbox"]
        component_mask = labeled[bbox] == int(meta["label"])
        area = float(meta["area"])
        elongation = float(meta["elongation"])
        should_watershed = area >= area_threshold or elongation >= elongation_threshold
        if should_watershed:
            local_labels, local_count = _watershed_single_component(
                component_mask,
                watershed_marker_threshold_px=watershed_marker_threshold_px,
                min_component_area_px=min_component_area_px,
            )
        else:
            local_labels, local_count = ndimage.label(component_mask, structure=np.ones((3, 3), dtype=np.uint8))
        if int(local_count) <= 0:
            continue
        for local_index, local_bbox in enumerate(ndimage.find_objects(local_labels), start=1):
            if local_bbox is None:
                continue
            local_component = local_labels[local_bbox] == local_index
            target = out[
                bbox[0].start + local_bbox[0].start : bbox[0].start + local_bbox[0].stop,
                bbox[1].start + local_bbox[1].start : bbox[1].start + local_bbox[1].stop,
            ]
            target[local_component] = next_label
            next_label += 1
    return out, int(np.max(out))


def _component_metadata(labeled: np.ndarray, min_component_area_px: int) -> list[dict[str, object]]:
    """Collect area and shape metadata for each connected component."""

    meta: list[dict[str, object]] = []
    for index, bbox in enumerate(ndimage.find_objects(labeled), start=1):
        if bbox is None:
            continue
        component = labeled[bbox] == index
        area = int(component.sum())
        if area < max(1, int(min_component_area_px)):
            continue
        height = int(bbox[0].stop - bbox[0].start)
        width = int(bbox[1].stop - bbox[1].start)
        elongation = float(max(width, height) / max(1, min(width, height)))
        meta.append(
            {
                "label": index,
                "bbox": bbox,
                "area": area,
                "elongation": elongation,
            }
        )
    return meta


def _resolve_selective_area_threshold(
    component_meta: list[dict[str, object]],
    min_component_area_px: int,
    watershed_selective_area_percentile: float,
) -> float:
    """Return the component area threshold above which watershed is applied."""

    if not component_meta:
        return float(max(1, int(min_component_area_px)))
    percentile = float(np.clip(float(watershed_selective_area_percentile), 0.0, 100.0))
    areas = np.asarray([float(item["area"]) for item in component_meta], dtype=float)
    return float(max(min_component_area_px, np.percentile(areas, percentile)))


def _watershed_single_component(
    component_mask: np.ndarray,
    watershed_marker_threshold_px: float | str,
    min_component_area_px: int,
) -> tuple[np.ndarray, int]:
    """Apply watershed inside one connected component only."""

    distance = ndimage.distance_transform_edt(component_mask)
    if float(distance.max()) <= 0.0:
        return ndimage.label(component_mask, structure=np.ones((3, 3), dtype=np.uint8))

    marker_threshold = _resolve_watershed_marker_threshold(
        component_mask,
        distance,
        watershed_marker_threshold_px=watershed_marker_threshold_px,
        min_component_area_px=min_component_area_px,
    )
    coordinates = peak_local_max(
        distance,
        labels=component_mask.astype(np.uint8),
        min_distance=1,
        threshold_abs=marker_threshold,
        footprint=np.ones((3, 3), dtype=bool),
        exclude_border=False,
    )
    if coordinates.size == 0:
        return ndimage.label(component_mask, structure=np.ones((3, 3), dtype=np.uint8))

    markers = np.zeros(component_mask.shape, dtype=np.int32)
    for marker_index, (y_coord, x_coord) in enumerate(coordinates, start=1):
        markers[int(y_coord), int(x_coord)] = marker_index
    labeled = watershed(-distance, markers, mask=component_mask)
    return np.asarray(labeled, dtype=np.int32), int(np.max(labeled))


def _resolve_watershed_marker_threshold(
    binary: np.ndarray,
    distance: np.ndarray,
    watershed_marker_threshold_px: float | str,
    min_component_area_px: int = 1,
) -> float:
    value = str(watershed_marker_threshold_px).strip().lower()
    if value and value != "auto":
        return max(0.5, float(value))
    return _estimate_watershed_marker_threshold(binary, distance, min_component_area_px=min_component_area_px)


def _estimate_watershed_marker_threshold(
    binary: np.ndarray,
    distance: np.ndarray,
    min_component_area_px: int = 1,
) -> float:
    labeled, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
    if count <= 0:
        return 1.5
    areas = np.bincount(labeled.ravel())[1:]
    valid_labels = [index + 1 for index, area in enumerate(areas) if area >= max(1, int(min_component_area_px))]
    if not valid_labels:
        valid_labels = [index + 1 for index, area in enumerate(areas) if area > 0]
    if not valid_labels:
        return 1.5
    peak_values = np.asarray([float(distance[labeled == label].max()) for label in valid_labels], dtype=float)
    if peak_values.size == 0:
        return 1.5
    reference_peak = float(np.percentile(peak_values, 35)) if peak_values.size >= 3 else float(np.median(peak_values))
    return float(np.clip(reference_peak * 0.75, 0.75, 24.0))


def detect_cells(
    animal_id: str,
    section_id: str,
    channel: str,
    image_channel: str,
    image_file: Path,
    registration_file: Path,
    mask: np.ndarray,
    intensity: np.ndarray,
    min_area_px: int,
    max_area_px: int = 0,
    apply_watershed: bool = False,
    watershed_marker_threshold_px: float | str = "auto",
    watershed_selective_area_percentile: float = 90.0,
    watershed_selective_elongation_threshold: float = 2.0,
    mask_source: str = "",
) -> list[DetectedObject]:
    """Detect connected components and compute basic intensity features."""

    labeled, _ = _label_connected_mask(
        mask,
        apply_watershed=apply_watershed,
        watershed_marker_threshold_px=watershed_marker_threshold_px,
        min_component_area_px=min_area_px,
        watershed_selective_area_percentile=watershed_selective_area_percentile,
        watershed_selective_elongation_threshold=watershed_selective_elongation_threshold,
    )
    object_slices = ndimage.find_objects(labeled)

    detected: list[DetectedObject] = []
    running_index = 1
    for label_index, bbox in enumerate(object_slices, start=1):
        if bbox is None:
            continue
        component = labeled[bbox] == label_index
        area_px = int(component.sum())
        if area_px < min_area_px:
            continue
        if int(max_area_px) > 0 and area_px > int(max_area_px):
            continue

        intensities = intensity[bbox][component]
        cy, cx = ndimage.center_of_mass(component)
        y0, x0 = bbox[0].start, bbox[1].start
        y1, x1 = bbox[0].stop, bbox[1].stop
        detected.append(
            DetectedObject(
                animal_id=animal_id,
                section_id=section_id,
                channel=channel,
                image_file=image_file,
                registration_file=registration_file,
                cell_id=f"{animal_id}_{section_id}_{channel}_{image_channel or 'NA'}_{running_index:06d}",
                centroid_x_px=float(x0 + cx),
                centroid_y_px=float(y0 + cy),
                area_px=int(area_px),
                mean_intensity=float(np.mean(intensities)) if intensities.size else 0.0,
                integrated_intensity=float(np.sum(intensities)) if intensities.size else 0.0,
                bbox=(x0, y0, x1, y1),
                mask_crop=component,
                bbox_origin=(x0, y0),
                image_channel=image_channel,
                mask_source=mask_source,
            )
        )
        running_index += 1
    return detected
