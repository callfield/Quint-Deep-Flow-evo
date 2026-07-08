"""Helpers for reusing existing Nutil outputs as compatibility references."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.repository import AtlasRepository
from data_models.models import DetectedObject, RegistrationSlice, SectionBundle
from registration.nonlinear import build_marker_inverse_warp, image_points_to_registration_source


@dataclass(slots=True)
class NutilReferenceArtifacts:
    """Paths to per-section Nutil outputs that can support compatibility exports."""

    object_report_path: Path | None = None
    region_report_path: Path | None = None
    atlas_image_path: Path | None = None

    @property
    def available(self) -> bool:
        """Whether at least one useful artifact exists."""

        return any(path is not None and path.exists() for path in self.paths())

    def paths(self) -> tuple[Path | None, Path | None, Path | None]:
        """Return all known artifact paths."""

        return self.object_report_path, self.region_report_path, self.atlas_image_path


def find_nutil_reference_artifacts(bundle: SectionBundle) -> NutilReferenceArtifacts:
    """Locate per-section Nutil reports / images for a discovered bundle."""

    if bundle.existing_result_dir is None or not bundle.existing_result_dir.exists():
        return NutilReferenceArtifacts()

    result_dir = bundle.existing_result_dir
    section_token = f"__{bundle.section_id}_"
    image_stem = bundle.image_path.stem

    return NutilReferenceArtifacts(
        object_report_path=_find_first(
            [
                result_dir / "Reports" / f"{bundle.animal_id}_{bundle.channel}_Objects" / f"{bundle.animal_id}_{bundle.channel}_Objects{section_token}.csv",
                *sorted((result_dir / "Reports").glob(f"**/*Objects*{section_token}.csv")),
            ]
        ),
        region_report_path=_find_first(
            [
                result_dir
                / "Reports"
                / f"{bundle.animal_id}_{bundle.channel}_RefAtlasRegions"
                / f"{bundle.animal_id}_{bundle.channel}_RefAtlasRegions{section_token}.csv",
                *sorted((result_dir / "Reports").glob(f"**/*RefAtlasRegions*{section_token}.csv")),
            ]
        ),
        atlas_image_path=_find_first(
            [
                *sorted((result_dir / "Images").glob(f"*_{image_stem}_Simple_Segmentation.png")),
                *sorted((result_dir / "Images").glob(f"*{bundle.section_id}*Simple*Segmentation*.png")),
            ]
        ),
    )


def load_nutil_object_report(path: Path) -> pd.DataFrame:
    """Load a Nutil object-level CSV into a normalized dataframe."""

    frame = _read_nutil_csv(path)
    required_columns = [
        "Object pixels",
        "Object area",
        "Center X",
        "Center Y",
        "Region ID",
        "Region Name",
        "Object ID",
    ]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in Nutil object report {path}")

    frame = frame[required_columns].copy()
    for column in ("Object pixels", "Object area", "Center X", "Center Y", "Region ID", "Object ID"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Center X", "Center Y", "Region ID"])
    frame["Region ID"] = frame["Region ID"].astype(int)
    frame["Object pixels"] = frame["Object pixels"].fillna(0).astype(int)
    frame["Object ID"] = frame["Object ID"].fillna(0).astype(int)
    frame["Region Name"] = frame["Region Name"].fillna("Unknown").astype(str)
    return frame.reset_index(drop=True)


def load_nutil_region_report(path: Path) -> pd.DataFrame:
    """Load a Nutil reference-atlas region CSV into a normalized dataframe."""

    frame = _read_nutil_csv(path)
    required_columns = [
        "Region ID",
        "Region Name",
        "Region pixels",
        "Region area",
        "Object count",
        "Object pixels",
        "Object area",
    ]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in Nutil region report {path}")

    frame = frame[required_columns].copy()
    for column in ("Region ID", "Region pixels", "Region area", "Object count", "Object pixels", "Object area"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Region ID"])
    frame["Region ID"] = frame["Region ID"].astype(int)
    frame["Region Name"] = frame["Region Name"].fillna("Unknown").astype(str)
    frame["Object count"] = frame["Object count"].fillna(0).astype(int)
    return frame.reset_index(drop=True)


def reference_objects_from_report(
    frame: pd.DataFrame,
    bundle: SectionBundle,
    registration_slice: RegistrationSlice,
    atlas: AtlasRepository,
    coordinate_unit: str,
    hemisphere_midline_threshold_um: float,
) -> list[DetectedObject]:
    """Convert a Nutil object CSV into DetectedObject-like rows for export."""

    from quantification.hemisphere import hemisphere_from_ml_um

    objects: list[DetectedObject] = []
    image_shape = (registration_slice.height, registration_slice.width)
    marker_warp = build_marker_inverse_warp(registration_slice, image_shape)

    for row in frame.to_dict("records"):
        centroid_x = float(row["Center X"])
        centroid_y = float(row["Center Y"])
        region_id = int(row["Region ID"])
        region = atlas.region_for_id(region_id)
        registration_xy = image_points_to_registration_source(
            np.asarray([[centroid_x, centroid_y]], dtype=np.float64),
            image_shape,
            registration_slice,
            mapper=marker_warp,
        )[0]
        quicknii = registration_slice.point_at_registration_pixel(registration_xy[0], registration_xy[1])
        allen_um = atlas.quicknii_to_allen_um(quicknii)
        ap, ml, dv = atlas.allen_um_to_bregma(allen_um, unit=coordinate_unit)
        ml_um = atlas.quicknii_ml_um(float(quicknii[0]))
        x0 = max(int(floor(centroid_x)), 0)
        y0 = max(int(floor(centroid_y)), 0)
        area_px = max(int(row["Object pixels"]), 1)
        objects.append(
            DetectedObject(
                animal_id=bundle.animal_id,
                section_id=bundle.section_id,
                channel=bundle.channel,
                image_file=bundle.image_path,
                registration_file=bundle.registration_json_path,
                cell_id=f"nutil_ref_{row['Object ID']}",
                centroid_x_px=centroid_x,
                centroid_y_px=centroid_y,
                area_px=area_px,
                mean_intensity=float("nan"),
                integrated_intensity=float("nan"),
                bbox=(x0, y0, x0 + 1, y0 + 1),
                mask_crop=np.ones((1, 1), dtype=bool),
                bbox_origin=(x0, y0),
                quicknii_xyz=tuple(float(value) for value in quicknii),
                allen_pir_um=tuple(float(value) for value in allen_um),
                ap=float(ap),
                ml=float(ml),
                dv=float(dv),
                hemisphere=hemisphere_from_ml_um(ml_um, hemisphere_midline_threshold_um),
                region_id=region_id,
                region_name=str(row["Region Name"]) if row["Region Name"] else (region.name if region else "Unknown"),
                parent_region_id=region.parent_id if region else None,
                hierarchy_ids=region.hierarchy_ids if region else [],
                hierarchy_names=region.hierarchy_names if region else [],
                assignment_fraction=1.0,
                assignment_method="nutil_reference",
            )
        )
    return objects


def reference_region_summary_rows(
    bundle: SectionBundle,
    region_frame: pd.DataFrame | None,
    reference_objects: list[DetectedObject],
) -> list[dict[str, object]]:
    """Build region summary rows from Nutil reference outputs."""

    rows: list[dict[str, object]] = []
    total_seen: set[int] = set()
    if region_frame is not None and not region_frame.empty:
        for row in region_frame.to_dict("records"):
            region_id = int(row["Region ID"])
            n_cells = int(row["Object count"])
            if n_cells <= 0 and region_id != 0:
                continue
            total_seen.add(region_id)
            region_area_um2 = float(row["Region area"]) if pd.notna(row["Region area"]) else None
            mean_area = None
            if n_cells > 0 and pd.notna(row["Object pixels"]):
                mean_area = float(row["Object pixels"]) / float(n_cells)
            density = None
            if region_area_um2 and region_area_um2 > 0 and n_cells > 0:
                density = float(n_cells / (region_area_um2 / 1_000_000.0))
            rows.append(
                {
                    "animal_id": bundle.animal_id,
                    "section_id": bundle.section_id,
                    "channel_or_combination": bundle.channel,
                    "region_id": region_id,
                    "region_name": row["Region Name"],
                    "hemisphere": "total",
                    "n_cells": n_cells,
                    "total_integrated_intensity": np.nan,
                    "mean_integrated_intensity": np.nan,
                    "mean_cell_area": mean_area,
                    "density_if_possible": density,
                    "region_area_um2": region_area_um2,
                    "summary_source": "nutil_reference",
                }
            )

    hemisphere_groups: dict[tuple[int, str], list[DetectedObject]] = {}
    for obj in reference_objects:
        key = (obj.region_id, obj.hemisphere)
        hemisphere_groups.setdefault(key, []).append(obj)
        if obj.region_id not in total_seen:
            total_key = (obj.region_id, "total")
            hemisphere_groups.setdefault(total_key, []).append(obj)

    for (region_id, hemisphere), objects in sorted(hemisphere_groups.items()):
        if hemisphere == "total":
            if region_id in total_seen:
                continue
        if region_id == 0 and hemisphere != "total":
            continue
        mean_area = float(np.mean([obj.area_px for obj in objects])) if objects else None
        rows.append(
            {
                "animal_id": bundle.animal_id,
                "section_id": bundle.section_id,
                "channel_or_combination": bundle.channel,
                "region_id": region_id,
                "region_name": objects[0].region_name,
                "hemisphere": hemisphere,
                "n_cells": len(objects),
                "total_integrated_intensity": np.nan,
                "mean_integrated_intensity": np.nan,
                "mean_cell_area": mean_area,
                "density_if_possible": None,
                "region_area_um2": None,
                "summary_source": "nutil_reference",
            }
        )
    return rows


def reference_section_summary(
    bundle: SectionBundle,
    reference_objects: list[DetectedObject],
    pixel_area_um2: float,
    qc_metrics: str,
) -> dict[str, object]:
    """Build a section summary using Nutil reference objects."""

    return {
        "animal_id": bundle.animal_id,
        "section_id": bundle.section_id,
        "channel": bundle.channel,
        "image_file": str(bundle.image_path),
        "json_file": str(bundle.registration_json_path),
        "n_detected_cells": len(reference_objects),
        "n_unassigned_cells": int(sum(1 for obj in reference_objects if obj.region_id == 0)),
        "left_count": int(sum(1 for obj in reference_objects if obj.hemisphere == "left")),
        "right_count": int(sum(1 for obj in reference_objects if obj.hemisphere == "right")),
        "pixel_area_um2": pixel_area_um2,
        "registration_qc_metrics": qc_metrics,
        "summary_source": "nutil_reference",
    }


def _find_first(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_nutil_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";", skiprows=1, engine="python")
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.dropna(axis=1, how="all")
    return frame.loc[:, ~frame.columns.str.contains(r"^Unnamed", regex=True)]
