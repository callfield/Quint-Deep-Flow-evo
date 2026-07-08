"""Region assignment helpers for detected cells."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from data_models.models import DetectedObject


class _AtlasLike(Protocol):
    def region_for_id(self, region_id: int): ...


def assign_object_region(
    obj: DetectedObject,
    region_map: np.ndarray,
    atlas: _AtlasLike,
    border_policy: str = "bigger",
) -> str:
    """Assign one atlas region to an object using the requested border policy."""

    def clear_assignment() -> None:
        obj.region_id = 0
        obj.region_name = "Unassigned"
        obj.parent_region_id = None
        obj.hierarchy_ids = []
        obj.hierarchy_names = []
        obj.assignment_fraction = 0.0

    x0, y0, x1, y1 = obj.bbox
    sub_region = region_map[y0:y1, x0:x1][obj.mask_crop]
    valid = sub_region[sub_region > 0]
    if valid.size == 0:
        clear_assignment()
        return "unassigned_no_region"

    labels, counts = np.unique(valid, return_counts=True)
    positive_labels = labels.astype(np.int64, copy=False)
    positive_counts = counts.astype(np.int64, copy=False)

    if positive_labels.size == 1:
        chosen_region_id = int(positive_labels[0])
        chosen_count = int(positive_counts[0])
        method = "single_region"
    else:
        normalized_policy = (border_policy or "bigger").strip().lower()
        if normalized_policy == "omit":
            clear_assignment()
            return "boundary_omit"
        if normalized_policy == "center":
            cx = int(round(obj.centroid_x_px))
            cy = int(round(obj.centroid_y_px))
            if 0 <= cy < region_map.shape[0] and 0 <= cx < region_map.shape[1]:
                center_region = int(region_map[cy, cx])
            else:
                center_region = 0
            if center_region > 0 and center_region in set(int(value) for value in positive_labels):
                chosen_region_id = center_region
                chosen_count = int(positive_counts[np.where(positive_labels == center_region)[0][0]])
                method = "boundary_center"
            else:
                clear_assignment()
                return "boundary_center_unassigned"
        else:
            majority_index = int(np.argmax(positive_counts))
            chosen_region_id = int(positive_labels[majority_index])
            chosen_count = int(positive_counts[majority_index])
            method = "boundary_bigger"

    region = atlas.region_for_id(chosen_region_id)
    obj.region_id = chosen_region_id
    obj.region_name = region.name if region else "Unknown"
    obj.parent_region_id = region.parent_id if region else None
    obj.hierarchy_ids = region.hierarchy_ids if region else []
    obj.hierarchy_names = region.hierarchy_names if region else []
    obj.assignment_fraction = float(chosen_count / max(valid.size, 1))
    return method
