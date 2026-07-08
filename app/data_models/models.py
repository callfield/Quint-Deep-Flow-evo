"""Core domain models used across the QUINTdeepflow application."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class AtlasRegion:
    """A single atlas region with hierarchy metadata."""

    region_id: int
    name: str
    color_rgb: tuple[int, int, int]
    parent_id: int | None = None
    children: list[int] = field(default_factory=list)
    hierarchy_ids: list[int] = field(default_factory=list)
    hierarchy_names: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RegistrationSlice:
    """Affine section placement from QuickNII / VisuAlign anchoring."""

    filename: str
    nr: int | None
    width: int
    height: int
    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    target_resolution: tuple[int, int, int] | None = None
    markers: list[list[float]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def point_at_pixel(
        self,
        x_px: float,
        y_px: float,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """Convert an image-space pixel into QuickNII atlas voxel coordinates."""

        out_h, out_w = image_shape
        x_weight = float(x_px) / float(max(out_w, 1))
        y_weight = float(y_px) / float(max(out_h, 1))
        return self.origin + (self.u * x_weight) + (self.v * y_weight)

    def point_at_registration_pixel(self, x_px: float, y_px: float) -> np.ndarray:
        """Convert a pixel in registration-image space into QuickNII atlas voxels."""

        return self.point_at_pixel(x_px, y_px, (max(self.height, 1), max(self.width, 1)))

    def section_pixel_area_um2(
        self,
        voxel_size_um: float,
    ) -> float:
        """Estimate section-plane area represented by one source pixel."""

        parallelogram_area_vox2 = np.linalg.norm(np.cross(self.u, self.v))
        pixel_count = max(self.width * self.height, 1)
        return float(parallelogram_area_vox2 / pixel_count) * (voxel_size_um**2)

    def scaled_for_target_resolution(
        self,
        target_resolution: tuple[int, int, int] | None,
    ) -> tuple["RegistrationSlice", tuple[float, float, float], bool]:
        """Return an anchoring-scaled copy if the atlas voxel grid changed."""

        source_resolution = self.target_resolution
        if (
            target_resolution is None
            or source_resolution is None
            or tuple(int(value) for value in source_resolution) == tuple(int(value) for value in target_resolution)
        ):
            return self, (1.0, 1.0, 1.0), False

        source = np.asarray(source_resolution, dtype=np.float64)
        target = np.asarray(target_resolution, dtype=np.float64)
        with np.errstate(divide="raise", invalid="raise"):
            scale = target / source
        scaled = replace(
            self,
            origin=self.origin.astype(np.float64, copy=False) * scale,
            u=self.u.astype(np.float64, copy=False) * scale,
            v=self.v.astype(np.float64, copy=False) * scale,
            target_resolution=tuple(int(value) for value in target_resolution),
        )
        return scaled, (float(scale[0]), float(scale[1]), float(scale[2])), True


@dataclass(slots=True)
class RegistrationData:
    """Parsed registration JSON container."""

    source_path: Path
    target: str
    target_resolution: tuple[int, int, int] | None
    slices: list[RegistrationSlice]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SectionBundle:
    """Resolved input files for a single animal / channel / section."""

    animal_id: str
    channel: str
    section_id: str
    image_path: Path
    mask_path: Path
    registration_json_path: Path
    registration_filename: str
    existing_result_dir: Path | None = None
    image_channel: str = ""
    mask_source: str = ""


@dataclass(slots=True)
class SectionGroup:
    """Same animal / section across one or more channels."""

    animal_id: str
    section_id: str
    bundles: list[SectionBundle] = field(default_factory=list)


@dataclass(slots=True)
class DetectedObject:
    """Single detected cell / object with registration and annotation metadata."""

    animal_id: str
    section_id: str
    channel: str
    image_file: Path
    registration_file: Path
    cell_id: str
    centroid_x_px: float
    centroid_y_px: float
    area_px: int
    mean_intensity: float
    integrated_intensity: float
    bbox: tuple[int, int, int, int]
    mask_crop: np.ndarray
    bbox_origin: tuple[int, int]
    quicknii_xyz: tuple[float, float, float] | None = None
    allen_pir_um: tuple[float, float, float] | None = None
    ap: float | None = None
    ml: float | None = None
    dv: float | None = None
    hemisphere: str = "unknown"
    region_id: int = 0
    region_name: str = "Unassigned"
    parent_region_id: int | None = None
    hierarchy_ids: list[int] = field(default_factory=list)
    hierarchy_names: list[str] = field(default_factory=list)
    assignment_fraction: float = 0.0
    overlap_group: str = ""
    matched_channels: list[str] = field(default_factory=list)
    overlap_group_id: str = ""
    atlas_display_code: int = 0
    atlas_patch_component: int = 0
    atlas_patch_id: str = ""
    assignment_method: str = "majority_region"
    image_channel: str = ""
    mask_source: str = ""


@dataclass(slots=True)
class OverlapGroupRecord:
    """One biological-cell group after cross-channel overlap resolution."""

    group_id: str
    animal_id: str
    section_id: str
    channels: tuple[str, ...]
    channel_flags: dict[str, int]
    objects: list[DetectedObject]
    region_id: int
    region_name: str
    hemisphere: str
    mean_intensity: float
    mean_cell_area: float


@dataclass(slots=True)
class SectionChannelResult:
    """Per-channel output for one section after quantification."""

    bundle: SectionBundle
    registration_slice: RegistrationSlice
    detected_objects: list[DetectedObject]
    region_map: np.ndarray
    hemisphere_map: np.ndarray
    overlay_preview_path: Path | None
    overlay_full_path: Path | None
    section_summary: dict[str, Any]
    numbered_atlas_preview_path: Path | None = None
    numbered_atlas_full_path: Path | None = None
    roi_legend_path: Path | None = None
    combined_overlay_preview_path: Path | None = None
    combined_overlay_full_path: Path | None = None
    source_image: np.ndarray | None = None
    reference_objects: list[DetectedObject] = field(default_factory=list)
    reference_section_summary: dict[str, Any] | None = None
    reference_region_rows: list[dict[str, Any]] = field(default_factory=list)
    summary_source: str = "native"
    warnings: list[str] = field(default_factory=list)
    overlap_groups: list[OverlapGroupRecord] = field(default_factory=list)


@dataclass(slots=True)
class PipelineOutput:
    """Pipeline return object with all exported tables."""

    output_dir: Path
    discovery_table: pd.DataFrame
    cell_level: pd.DataFrame
    region_summary: pd.DataFrame
    section_summary: pd.DataFrame
    multichannel_summary: pd.DataFrame
    processing_log: pd.DataFrame
    comparison_report: pd.DataFrame
