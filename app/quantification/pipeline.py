"""Main quantification pipeline orchestration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import json
import logging
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import ndimage

from atlas.display_codes import build_display_codebook
from atlas.repository import AtlasRepository
from atlas.section_cache import (
    RegisteredSectionCacheEntry,
    build_registered_section_cache_key,
    build_registered_section_cache_path,
    load_registered_section_cache,
    save_registered_section_cache,
)
from config.settings import AppConfig, save_app_config
from data_models.models import PipelineOutput, RegistrationSlice, SectionBundle, SectionChannelResult, SectionGroup
from exporters.comparison import generate_comparison_report
from exporters.csv_exporter import cell_rows_from_objects, metrics_json, write_tables
from io_utils.discovery import discovery_to_dataframe, discover_section_groups
from io_utils.image_io import (
    grayscale_intensity,
    load_image_array,
    load_mask_array,
    normalize_ilastik_mask_file_inplace,
)
from io_utils.nutil_reference import (
    find_nutil_reference_artifacts,
    load_nutil_object_report,
    load_nutil_region_report,
    reference_objects_from_report,
    reference_region_summary_rows,
    reference_section_summary,
)
from multichannel.matcher import apply_multichannel_matching, expand_overlap_set_spec
from overlays.render import (
    build_registered_maps,
    qdf2d_base_registration_slice,
    qdf2d_transform_payload,
    save_multichannel_overlay_images,
)
from quantification.assignment import assign_object_region
from quantification.detector import detect_cells
from quantification.evo_omit import apply_qdf1_evo_omit_to_results
from quantification.hemisphere import hemisphere_from_ml_um
from quantification.section_summary import build_section_region_summary
from registration.nonlinear import build_marker_inverse_warp, build_piecewise_affine_mapper, image_points_to_registration_source
from registration.parser import match_registration_slice, parse_registration_file

LOG = logging.getLogger(__name__)


class _CsvRowSpool:
    """Append row dictionaries to a temporary CSV without retaining all rows in memory."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.columns: list[str] = []
        self.row_count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append_rows(self, rows: list[dict[str, object]]) -> None:
        if not rows:
            return
        frame = pd.DataFrame(rows)
        if self.row_count == 0:
            self.columns = list(frame.columns)
            frame.to_csv(self.path, index=False)
        else:
            for column in self.columns:
                if column not in frame.columns:
                    frame[column] = pd.NA
            frame = frame.reindex(columns=self.columns)
            frame.to_csv(self.path, mode="a", header=False, index=False)
        self.row_count += int(len(frame))
        del frame

    def materialize(self, output_path: Path) -> pd.DataFrame:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.replace(output_path)
        else:
            pd.DataFrame(columns=self.columns).to_csv(output_path, index=False)
        return pd.DataFrame(columns=self.columns)


def _drop_row_keys(rows: list[dict[str, object]], keys: set[str]) -> list[dict[str, object]]:
    """Return rows with transient/internal keys removed before CSV spooling."""

    if not rows:
        return rows
    return [{key: value for key, value in row.items() if key not in keys} for row in rows]


def _display_code_map_from_region_maps(region_map: np.ndarray, hemisphere_map: np.ndarray) -> np.ndarray:
    """Build the signed display-code plane used by native overlays."""

    display_map = region_map.astype(np.int32, copy=True)
    display_map[hemisphere_map < 0] *= -1
    display_map[hemisphere_map == 0] = 0
    return display_map


def _format_atlas_patch_id(animal_id: str, section_id: str, display_code: int, component_label: int) -> str:
    """Create a stable per-slice patch identifier."""

    return f"{animal_id}::{section_id}::{int(display_code)}::{int(component_label)}"


class QuantificationPipeline:
    """End-to-end discovery, quantification, export, and comparison runner."""

    def __init__(self, config: AppConfig, logger: logging.Logger | None = None) -> None:
        self.config = config
        self.logger = logger or LOG

    def run(
        self,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> PipelineOutput:
        """Run the full pipeline and export CSV outputs."""

        groups = discover_section_groups(self.config)
        discovery_df = discovery_to_dataframe(groups)
        output_dir = self.config.output.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        # Keep registered section atlas caches inside each Results folder so the
        # run is portable across PCs together with its own reusable cache.
        self.config.atlas.registered_section_cache_dir = output_dir / "atlas_cache" / "registered_sections"
        mask_normalization_df = pd.DataFrame()
        if self.config.processing.normalize_ilastik_masks_to_binary:
            if progress_callback:
                progress_callback("Normalizing ilastik masks", 0.02)
            mask_normalization_df = self._normalize_ilastik_masks(groups)
        atlas = AtlasRepository(self.config.atlas)
        atlas.load()
        display_codebook_df = build_display_codebook(atlas.regions)

        if self.config.output.save_resolved_config:
            save_app_config(self.config, output_dir / "resolved_config.yaml")

        spool_dir = output_dir / "_qdf_tmp_rows"
        if spool_dir.exists():
            shutil.rmtree(spool_dir, ignore_errors=True)
        spool_dir.mkdir(parents=True, exist_ok=True)
        cell_spool = _CsvRowSpool(spool_dir / "cell_level.csv")
        region_spool = _CsvRowSpool(spool_dir / "region_summary.csv")
        section_spool = _CsvRowSpool(spool_dir / "section_channel_summary.csv")

        processing_rows: list[dict[str, object]] = []
        comparison_inputs: list[SectionChannelResult] = []
        animal_multichannel_channel_maps: dict[str, pd.DataFrame] = {}

        def consume_payload(payload: dict[str, object]) -> None:
            processing_rows.extend(payload["processing_rows"])  # type: ignore[arg-type]
            channel_map_frame = payload.get("channel_map_frame")
            animal_id = str(payload.get("animal_id", ""))
            if isinstance(channel_map_frame, pd.DataFrame) and animal_id and animal_id not in animal_multichannel_channel_maps:
                animal_multichannel_channel_maps[animal_id] = channel_map_frame.copy()
            cell_spool.append_rows(payload.get("cell_rows", []))  # type: ignore[arg-type]
            region_spool.append_rows(
                _drop_row_keys(
                    payload.get("region_rows", []),  # type: ignore[arg-type]
                    {"channel", "summary_source"},
                )
            )
            section_spool.append_rows(
                _drop_row_keys(
                    payload.get("section_rows", []),  # type: ignore[arg-type]
                    {"channel", "image_file", "json_file", "mask_source", "registration_qc_metrics", "summary_source"},
                )
            )
            if self.config.comparison.enabled:
                comparison_inputs.extend(payload.get("comparison_results", []))  # type: ignore[arg-type]
            payload.clear()
            gc.collect()

        total_groups = max(len(groups), 1)
        parallel_workers = self._resolve_parallel_workers(len(groups))
        if parallel_workers <= 1:
            for index, group in enumerate(groups, start=1):
                if progress_callback:
                    progress_callback(f"Processing {group.animal_id} {group.section_id}", (index - 1) / total_groups)
                payload = self._process_group(group, atlas, output_dir)
                consume_payload(payload)
                if progress_callback:
                    progress_callback(f"Processed {group.animal_id} {group.section_id}", index / total_groups)
        else:
            with ThreadPoolExecutor(max_workers=parallel_workers, thread_name_prefix="quintnext") as executor:
                future_map = {
                    executor.submit(self._process_group, group, atlas, output_dir): (index, group)
                    for index, group in enumerate(groups, start=1)
                }
                completed = 0
                for future in as_completed(future_map):
                    index, group = future_map[future]
                    payload = future.result()
                    payload["group_index"] = index
                    consume_payload(payload)
                    completed += 1
                    if progress_callback:
                        progress_callback(
                            f"Processed {group.animal_id} {group.section_id} [{completed}/{len(groups)}]",
                            completed / total_groups,
                        )

        if progress_callback and self.config.comparison.enabled:
            progress_callback("Comparing with existing Nutil outputs", 0.96)

        comparison_df = (
            generate_comparison_report(comparison_inputs, atlas, self.config.comparison, output_dir)
            if self.config.comparison.enabled
            else pd.DataFrame()
        )
        processing_df = pd.DataFrame(processing_rows)

        tables = {
            "discovery_table.csv": discovery_df,
            "atlas_display_codebook.csv": display_codebook_df,
            "processing_log.csv": processing_df,
        }
        if not mask_normalization_df.empty:
            tables["mask_normalization_report.csv"] = mask_normalization_df
        if self.config.comparison.enabled:
            tables["comparison_report.csv"] = comparison_df

        spooled_table_names = {"cell_level.csv", "region_summary.csv", "section_channel_summary.csv", "section_summary.csv"}
        self._cleanup_obsolete_outputs(output_dir, set(tables) | spooled_table_names, comparison_enabled=self.config.comparison.enabled)
        write_tables(output_dir, tables)
        cell_df = cell_spool.materialize(output_dir / "cell_level.csv")
        region_spool.materialize(output_dir / "region_summary.csv")
        section_spool.materialize(output_dir / "section_channel_summary.csv")
        region_df = pd.read_csv(output_dir / "region_summary.csv") if (output_dir / "region_summary.csv").exists() else pd.DataFrame()
        section_channel_df = (
            pd.read_csv(output_dir / "section_channel_summary.csv")
            if (output_dir / "section_channel_summary.csv").exists()
            else pd.DataFrame()
        )
        section_df = build_section_region_summary(region_df, section_channel_df)
        section_df.to_csv(output_dir / "section_summary.csv", index=False)
        shutil.rmtree(spool_dir, ignore_errors=True)
        animal_channel_map_paths = self._write_animal_channel_map_workbooks(
            output_dir,
            animal_multichannel_channel_maps,
        )
        if self._qdf1_omit_import_enabled():
            omit_result = apply_qdf1_evo_omit_to_results(
                output_dir=output_dir,
                registration_json_path=self.config.discovery.atlas_json_path,
                logger=self.logger,
            )
            if omit_result.get("enabled"):
                self.logger.info("Applied QDF1 omit import: %s", omit_result)
                if (output_dir / "region_summary.csv").exists():
                    region_df = pd.read_csv(output_dir / "region_summary.csv")
                if (output_dir / "section_summary.csv").exists():
                    section_df = pd.read_csv(output_dir / "section_summary.csv")
            else:
                self.logger.info("Skipped QDF1 omit import: %s", omit_result)
        if progress_callback:
            progress_callback("Done", 1.0)

        return PipelineOutput(
            output_dir=output_dir,
            discovery_table=discovery_df,
            cell_level=cell_df,
            region_summary=region_df,
            section_summary=section_df,
            multichannel_summary=pd.DataFrame(),
            processing_log=processing_df,
            comparison_report=comparison_df,
        )

    def _resolve_parallel_workers(self, group_count: int) -> int:
        """Return the number of worker threads to use for section-group processing."""

        if group_count <= 1:
            return 1
        requested = int(getattr(self.config.processing, "parallel_workers", 0) or 0)
        if requested > 0:
            return max(1, min(requested, group_count))
        auto_workers = max(1, (os.cpu_count() or 2) // 2)
        return max(1, min(auto_workers, group_count, 8))

    def _qdf1_omit_import_enabled(self) -> bool:
        """Return True when QDFevo_2_AtlasFitter omit masks should update QDF2 outputs."""

        for env_name in ("QUINTDEEPFLOW_IMPORT_QDF1_OMIT", "QUINTDEEPFLOW_IMPORT_QDF1_EVO_OMIT"):
            if str(os.environ.get(env_name, "")).strip().lower() in {"1", "true", "yes", "on"}:
                return True
        return False

    def _process_group(
        self,
        group: SectionGroup,
        atlas: AtlasRepository,
        output_dir: Path,
    ) -> dict[str, object]:
        """Process all bundles for one animal/section group, optionally producing multichannel outputs."""

        processing_rows: list[dict[str, object]] = []
        group_results: list[SectionChannelResult] = []
        channel_map_frame: pd.DataFrame | None = None
        registered_map_cache: dict[str, RegisteredSectionCacheEntry] = {}
        registered_maps_by_shape: dict[tuple[int, int], RegisteredSectionCacheEntry] = {}

        registration_bundle = self._select_registration_bundle(group.bundles)
        shared_registration_error = ""
        shared_registration_slice = None
        if registration_bundle is not None:
            try:
                registration = parse_registration_file(registration_bundle.registration_json_path)
                shared_registration_slice = match_registration_slice(
                    registration,
                    registration_bundle.registration_filename,
                )
            except Exception as exc:
                shared_registration_error = str(exc)
                self.logger.exception(
                    "Failed loading registration bundle %s %s %s [%s]",
                    registration_bundle.animal_id,
                    registration_bundle.section_id,
                    registration_bundle.channel,
                    registration_bundle.image_channel,
                )

        for bundle in group.bundles:
            started = time.perf_counter()
            warning_text = ""
            error_text = ""
            try:
                if shared_registration_slice is None or registration_bundle is None:
                    raise RuntimeError(
                        shared_registration_error
                        or f"No registration bundle available for {group.animal_id} {group.section_id}"
                    )
                result = self._process_bundle(
                    bundle,
                    atlas,
                    output_dir,
                    registration_slice=shared_registration_slice,
                    registration_bundle=registration_bundle,
                    registered_map_cache=registered_map_cache,
                    registered_maps_by_shape=registered_maps_by_shape,
                )
                group_results.append(result)
                warning_text = ";".join(result.warnings)
                qc_metrics = json.loads(str(result.section_summary.get("registration_qc_metrics", "{}")))
                atlas_map_cache_source = str(qc_metrics.get("atlas_map_cache_source", ""))
                atlas_map_elapsed_seconds = qc_metrics.get("atlas_map_elapsed_seconds", "")
                atlas_map_cache_path = str(qc_metrics.get("atlas_map_cache_path", ""))
            except Exception as exc:
                error_text = str(exc)
                atlas_map_cache_source = ""
                atlas_map_elapsed_seconds = ""
                atlas_map_cache_path = ""
                self.logger.exception("Failed processing %s %s %s", bundle.animal_id, bundle.section_id, bundle.channel)
            finally:
                processing_rows.append(
                    {
                        "animal_id": bundle.animal_id,
                        "section_id": bundle.section_id,
                        "channel": bundle.channel,
                        "image_channel": bundle.image_channel,
                        "image_file": str(bundle.image_path),
                        "mask_file": str(bundle.mask_path),
                        "mask_source": bundle.mask_source,
                        "registration_json": str(bundle.registration_json_path),
                        "registration_channel": registration_bundle.image_channel if registration_bundle else "",
                        "success": error_text == "",
                        "warning": warning_text,
                        "error": error_text,
                        "atlas_map_cache_source": atlas_map_cache_source,
                        "atlas_map_elapsed_seconds": atlas_map_elapsed_seconds,
                        "atlas_map_cache_path": atlas_map_cache_path,
                        "elapsed_seconds": round(time.perf_counter() - started, 4),
                    }
                )

        if group_results:
            apply_multichannel_matching(group_results, self.config.matching)
            if self.config.processing.overlay_enabled and self.config.processing.combined_overlay_enabled:
                try:
                    combined_assets = self._save_group_overlay(group_results, atlas, output_dir)
                    if isinstance(combined_assets.get("channel_map_frame"), pd.DataFrame):
                        channel_map_frame = combined_assets["channel_map_frame"].copy()
                    self._attach_combined_overlay_paths(group_results, combined_assets)
                except Exception:
                    self.logger.exception("Failed saving multichannel overlay for %s %s", group.animal_id, group.section_id)
                    for result in group_results:
                        result.warnings.append("combined_overlay_failed")

        analysis_channels = self._analysis_channels(group_results)
        cell_rows: list[dict[str, object]] = []
        region_rows: list[dict[str, object]] = []
        section_rows: list[dict[str, object]] = []
        comparison_results: list[SectionChannelResult] = []
        for result in group_results:
            cell_rows.extend(
                cell_rows_from_objects(
                    result.detected_objects,
                    self.config.processing.coordinate_unit,
                    analysis_channels,
                    include_patch_ids=bool(getattr(self.config.processing, "export_patch_ids", False)),
                )
            )
            region_rows.extend(self._region_summary_rows(result))
            section_rows.append(dict(result.section_summary))
            if self.config.comparison.enabled:
                comparison_results.append(result)

        self._release_heavy_group_results(
            group_results,
            keep_detected_objects=bool(self.config.comparison.enabled),
        )

        return {
            "group_index": 0,
            "animal_id": group.animal_id,
            "section_id": group.section_id,
            "processing_rows": processing_rows,
            "cell_rows": cell_rows,
            "region_rows": region_rows,
            "section_rows": section_rows,
            "comparison_results": comparison_results,
            "channel_map_frame": channel_map_frame,
        }

    def _release_heavy_group_results(
        self,
        group_results: list[SectionChannelResult],
        *,
        keep_detected_objects: bool,
    ) -> None:
        """Drop per-section image/map/mask arrays once CSV rows and overlays are built."""

        empty_region_map = np.empty((0, 0), dtype=np.uint32)
        empty_hemisphere_map = np.empty((0, 0), dtype=np.int8)
        empty_mask = np.zeros((0, 0), dtype=bool)

        for result in group_results:
            result.source_image = None
            result.region_map = empty_region_map
            result.hemisphere_map = empty_hemisphere_map
            self._strip_object_mask_crops(result.detected_objects, empty_mask)
            self._strip_object_mask_crops(result.reference_objects, empty_mask)
            for overlap_group in result.overlap_groups:
                self._strip_object_mask_crops(overlap_group.objects, empty_mask)
            result.overlap_groups.clear()
            result.reference_objects.clear()
            result.reference_region_rows.clear()
            if not keep_detected_objects:
                result.detected_objects.clear()

        if not keep_detected_objects:
            group_results.clear()
        gc.collect()

    @staticmethod
    def _strip_object_mask_crops(objects: list, empty_mask: np.ndarray) -> None:
        """Replace object-level ROI masks with one shared empty mask."""

        for obj in objects:
            obj.mask_crop = empty_mask

    def _process_bundle(
        self,
        bundle: SectionBundle,
        atlas: AtlasRepository,
        output_dir: Path,
        registration_slice: RegistrationSlice,
        registration_bundle: SectionBundle,
        registered_map_cache: dict[str, RegisteredSectionCacheEntry],
        registered_maps_by_shape: dict[tuple[int, int], RegisteredSectionCacheEntry],
    ) -> SectionChannelResult:
        image = load_image_array(bundle.image_path, grayscale=False)
        intensity = grayscale_intensity(image)
        mask = load_mask_array(bundle.mask_path, threshold=self._mask_threshold_for_bundle(bundle))
        reg_slice, registration_scale_xyz, registration_scaled = atlas.adapted_registration_slice(registration_slice)

        detected = detect_cells(
            animal_id=bundle.animal_id,
            section_id=bundle.section_id,
            channel=bundle.channel,
            image_channel=bundle.image_channel,
            image_file=bundle.image_path,
            registration_file=registration_bundle.registration_json_path,
            mask=mask,
            intensity=intensity,
            min_area_px=self._min_area_for_bundle(bundle),
            max_area_px=self._max_area_for_bundle(bundle),
            apply_watershed=self._apply_watershed_for_bundle(bundle),
            watershed_marker_threshold_px=self._watershed_marker_threshold_for_bundle(bundle),
            watershed_selective_area_percentile=self._watershed_selective_area_percentile_for_bundle(bundle),
            watershed_selective_elongation_threshold=self._watershed_selective_elongation_threshold_for_bundle(bundle),
            mask_source=bundle.mask_source,
        )
        del mask, intensity

        output_shape = tuple(int(value) for value in image.shape[:2])
        if output_shape in registered_maps_by_shape:
            cache_entry = registered_maps_by_shape[output_shape]
            atlas_map_source = "memory_same_shape"
            atlas_map_elapsed = 0.0
        else:
            cache_entry, atlas_map_source, atlas_map_elapsed = self._resolve_registered_maps(
                bundle=bundle,
                atlas=atlas,
                registration_slice=reg_slice,
                registration_bundle=registration_bundle,
                output_shape=output_shape,
                registered_map_cache=registered_map_cache,
            )
            registered_maps_by_shape[output_shape] = cache_entry
        region_map = cache_entry.region_map
        hemisphere_map = cache_entry.hemisphere_map
        qc_metrics = dict(cache_entry.qc_metrics)
        qc_metrics["atlas_map_cache_source"] = atlas_map_source
        qc_metrics["atlas_map_elapsed_seconds"] = round(float(atlas_map_elapsed), 4)
        qc_metrics["atlas_map_cache_path"] = str(cache_entry.cache_path) if cache_entry.cache_path else ""
        qdf2d_payload = qdf2d_transform_payload(reg_slice)
        coordinate_slice = reg_slice
        marker_warp = build_marker_inverse_warp(reg_slice, image.shape[:2])
        if qdf2d_payload is not None:
            coordinate_slice = qdf2d_base_registration_slice(reg_slice, qdf2d_payload)
            corner_rows = qdf2d_payload.get("corner_markers")
            marker_rows = corner_rows if isinstance(corner_rows, list) else []
            marker_rows = [list(row) for row in marker_rows if isinstance(row, list) and len(row) >= 4]
            marker_rows.extend([list(marker) for marker in reg_slice.markers if len(marker) >= 4])
            marker_warp = build_piecewise_affine_mapper(reg_slice.width, reg_slice.height, marker_rows)

        if detected:
            centroids_xy = np.asarray([[obj.centroid_x_px, obj.centroid_y_px] for obj in detected], dtype=np.float64)
            registration_xy = image_points_to_registration_source(
                centroids_xy,
                image.shape[:2],
                coordinate_slice,
                mapper=marker_warp,
            )
            registration_height = float(max(coordinate_slice.height, 1))
            registration_width = float(max(coordinate_slice.width, 1))
            centroid_quicknii = (
                coordinate_slice.origin.astype(np.float64, copy=False)[None, :]
                + coordinate_slice.u.astype(np.float64, copy=False)[None, :] * (registration_xy[:, 0:1] / registration_width)
                + coordinate_slice.v.astype(np.float64, copy=False)[None, :] * (registration_xy[:, 1:2] / registration_height)
            )
            allen_um = atlas.quicknii_to_allen_um(centroid_quicknii)
            ap_ml_dv = atlas.allen_um_to_bregma_array(allen_um, unit=self.config.processing.coordinate_unit)
            ml_um = atlas.quicknii_ml_um_array(centroid_quicknii[:, 0])

            for index, obj in enumerate(detected):
                obj.quicknii_xyz = tuple(float(value) for value in centroid_quicknii[index])
                obj.allen_pir_um = tuple(float(value) for value in allen_um[index])
                obj.ap = float(ap_ml_dv[index, 0])
                obj.ml = float(ap_ml_dv[index, 1])
                obj.dv = float(ap_ml_dv[index, 2])
                obj.hemisphere = hemisphere_from_ml_um(float(ml_um[index]), self.config.processing.hemisphere_midline_threshold_um)
                region_method = assign_object_region(
                    obj,
                    region_map,
                    atlas,
                    border_policy=self.config.processing.border_assignment_policy,
                )
                if qdf2d_payload is not None:
                    obj.assignment_method = f"qdf2d_layer:{region_method}"
                elif marker_warp is not None:
                    obj.assignment_method = f"visualign_piecewise_affine:{region_method}"
                else:
                    obj.assignment_method = region_method
        self._annotate_object_patch_metadata(bundle, detected, region_map, hemisphere_map)

        warnings: list[str] = []
        reference_objects = []
        reference_region_rows: list[dict[str, object]] = []
        reference_section_summary_row: dict[str, object] | None = None
        summary_source = "native"
        artifacts = find_nutil_reference_artifacts(bundle) if self.config.reference.enabled else None

        if artifacts and artifacts.object_report_path is not None:
            try:
                reference_object_frame = load_nutil_object_report(artifacts.object_report_path)
                reference_objects = reference_objects_from_report(
                    reference_object_frame,
                    bundle=bundle,
                    registration_slice=reg_slice,
                    atlas=atlas,
                    coordinate_unit=self.config.processing.coordinate_unit,
                    hemisphere_midline_threshold_um=self.config.processing.hemisphere_midline_threshold_um,
                )
                reference_region_frame = (
                    load_nutil_region_report(artifacts.region_report_path)
                    if artifacts.region_report_path is not None
                    else None
                )
                reference_region_rows = reference_region_summary_rows(
                    bundle=bundle,
                    region_frame=reference_region_frame,
                    reference_objects=reference_objects,
                )
            except Exception:
                warnings.append("reference_artifact_parse_failed")
                self.logger.exception(
                    "Failed loading Nutil reference artifacts for %s %s %s",
                    bundle.animal_id,
                    bundle.section_id,
                    bundle.channel,
                )

        overlay_preview_path = None
        overlay_full_path = None
        numbered_atlas_preview_path = None
        numbered_atlas_full_path = None
        roi_legend_path = None
        overlay_source = "native_reslice"

        qc_metric_payload = {
            **qc_metrics,
            "marker_count": len(reg_slice.markers),
            "registration_scaled_for_atlas": registration_scaled,
            "registration_scale_xyz": [float(value) for value in registration_scale_xyz],
            "registration_source_target_resolution": list(registration_slice.target_resolution or ()),
            "atlas_target_resolution": list(int(value) for value in atlas.require_labels().shape),
            "atlas_display_codebook": str(output_dir / "atlas_display_codebook.csv"),
            "overlay_full": str(overlay_full_path) if overlay_full_path else "",
            "overlay_source": overlay_source,
            "numbered_atlas_full": str(numbered_atlas_full_path) if numbered_atlas_full_path else "",
            "reference_object_report": str(artifacts.object_report_path) if artifacts and artifacts.object_report_path else "",
            "reference_region_report": str(artifacts.region_report_path) if artifacts and artifacts.region_report_path else "",
            "reference_atlas_image": str(artifacts.atlas_image_path) if artifacts and artifacts.atlas_image_path else "",
            "registration_bundle": registration_bundle.channel,
            "registration_channel": registration_bundle.image_channel,
            "mask_source": bundle.mask_source,
        }
        section_summary = {
            "animal_id": bundle.animal_id,
            "section_id": bundle.section_id,
            "channel": bundle.channel,
            "image_channel": bundle.image_channel,
            "registration_image_channel": registration_bundle.image_channel,
            "image_file": str(bundle.image_path),
            "json_file": str(registration_bundle.registration_json_path),
            "mask_source": bundle.mask_source,
            "n_detected_cells": len(detected),
            "n_unassigned_cells": int(sum(1 for obj in detected if obj.region_id == 0)),
            "left_count": int(sum(1 for obj in detected if obj.hemisphere == "left")),
            "right_count": int(sum(1 for obj in detected if obj.hemisphere == "right")),
            "pixel_area_um2": qc_metrics["pixel_area_um2"],
            "registration_qc_metrics": metrics_json(qc_metric_payload),
            "summary_source": "native",
        }

        if reference_objects:
            reference_section_summary_row = reference_section_summary(
                bundle=bundle,
                reference_objects=reference_objects,
                pixel_area_um2=qc_metrics["pixel_area_um2"],
                qc_metrics=metrics_json(qc_metric_payload),
            )
            if self.config.reference.prefer_reference_summary_if_available:
                summary_source = "nutil_reference"
                warnings.append("reference_summary_primary")

        if qc_metrics["atlas_coverage_fraction"] < 0.5:
            warnings.append("low_atlas_coverage")

        return SectionChannelResult(
            bundle=bundle,
            registration_slice=reg_slice,
            detected_objects=detected,
            region_map=region_map,
            hemisphere_map=hemisphere_map,
            overlay_preview_path=overlay_preview_path,
            overlay_full_path=overlay_full_path,
            section_summary=section_summary,
            numbered_atlas_preview_path=numbered_atlas_preview_path,
            numbered_atlas_full_path=numbered_atlas_full_path,
            roi_legend_path=roi_legend_path,
            source_image=image,
            reference_objects=reference_objects,
            reference_section_summary=reference_section_summary_row,
            reference_region_rows=reference_region_rows,
            summary_source=summary_source,
            warnings=warnings,
        )

    def _region_summary_rows(self, result: SectionChannelResult) -> list[dict[str, object]]:
        pixel_area_um2 = float(result.section_summary["pixel_area_um2"])
        region_map = result.region_map
        hemisphere_map = result.hemisphere_map
        overlap_metrics = self._overlap_region_metrics(result.overlap_groups)
        overlap_defaults = self._overlap_metric_defaults()

        objects_by_key: dict[tuple[int, str], list] = defaultdict(list)
        for obj in result.detected_objects:
            objects_by_key[(obj.region_id, obj.hemisphere)].append(obj)
            objects_by_key[(obj.region_id, "total")].append(obj)

        region_rows: list[dict[str, object]] = []
        valid_region = region_map > 0
        total_region_ids, total_counts = np.unique(region_map[valid_region], return_counts=True)
        area_lookup: dict[tuple[int, str], float] = {
            (int(region_id), "total"): float(count) * pixel_area_um2
            for region_id, count in zip(total_region_ids, total_counts, strict=False)
        }
        for hemisphere_name, hemisphere_code in (("left", -1), ("right", 1)):
            mask = valid_region & (hemisphere_map == hemisphere_code)
            region_ids, counts = np.unique(region_map[mask], return_counts=True)
            for region_id, count in zip(region_ids, counts, strict=False):
                area_lookup[(int(region_id), hemisphere_name)] = float(count) * pixel_area_um2

        for (region_id, hemisphere), objects in sorted(objects_by_key.items()):
            if region_id == 0:
                continue
            total_integrated = float(sum(obj.integrated_intensity for obj in objects))
            mean_integrated = float(np.mean([obj.mean_intensity for obj in objects])) if objects else 0.0
            mean_area = float(np.mean([obj.area_px for obj in objects])) if objects else 0.0
            patch_summary = self._region_patch_summary(objects)
            area_um2 = area_lookup.get((region_id, hemisphere), 0.0)
            density = float(len(objects) / (area_um2 / 1_000_000.0)) if area_um2 > 0 else None
            row = {
                "animal_id": result.bundle.animal_id,
                "section_id": result.bundle.section_id,
                "channel": result.bundle.channel,
                "image_channel": result.bundle.image_channel,
                "channel_or_combination": self._bundle_output_name(result.bundle),
                "region_id": region_id,
                "region_name": objects[0].region_name,
                "hemisphere": hemisphere,
                "n_cells": len(objects),
                "total_integrated_intensity": total_integrated,
                "mean_integrated_intensity": mean_integrated,
                "mean_cell_area": mean_area,
                "density_if_possible": density,
                "region_area_um2": area_um2,
                "summary_source": "native",
                "atlas_patch_count": patch_summary["atlas_patch_count"],
                "atlas_display_codes": patch_summary["atlas_display_codes"],
                "atlas_patch_components": patch_summary["atlas_patch_components"],
                "atlas_patch_ids": patch_summary["atlas_patch_ids"],
            }
            row.update(overlap_defaults)
            row.update(overlap_metrics.get((region_id, hemisphere), {}))
            region_rows.append(row)
        return region_rows

    def _region_patch_summary(self, objects: list) -> dict[str, object]:
        """Summarize atlas patch membership for one region-summary row."""

        display_codes = sorted(
            {
                int(getattr(obj, "atlas_display_code", 0))
                for obj in objects
                if int(getattr(obj, "atlas_display_code", 0)) != 0
            }
        )
        patch_components = sorted(
            {
                f"{int(getattr(obj, 'atlas_display_code', 0))}:{int(getattr(obj, 'atlas_patch_component', 0))}"
                for obj in objects
                if int(getattr(obj, "atlas_display_code", 0)) != 0 and int(getattr(obj, "atlas_patch_component", 0)) > 0
            }
        )
        patch_ids = sorted(
            {
                str(getattr(obj, "atlas_patch_id", "")).strip()
                for obj in objects
                if str(getattr(obj, "atlas_patch_id", "")).strip()
            }
        )
        return {
            "atlas_patch_count": int(len(patch_ids)),
            "atlas_display_codes": ";".join(str(code) for code in display_codes),
            "atlas_patch_components": ";".join(patch_components),
            "atlas_patch_ids": ";".join(patch_ids),
        }

    def _analysis_channels(self, section_results: list[SectionChannelResult]) -> list[str]:
        configured = [channel.upper() for channel in self.config.processing.analysis_image_channels if channel and channel.strip()]
        if configured:
            return configured
        derived = {
            (result.bundle.image_channel or result.bundle.channel).upper()
            for result in section_results
            if (result.bundle.image_channel or result.bundle.channel)
        }
        return sorted(derived, key=str.upper)

    def _parsed_overlap_specs(self) -> list[tuple[str, tuple[tuple[str, bool], ...]]]:
        parsed: list[tuple[str, tuple[tuple[str, bool], ...]]] = []
        seen: set[str] = set()
        for raw_spec in self.config.matching.combinations:
            if not raw_spec or not str(raw_spec).strip():
                continue
            try:
                expanded = expand_overlap_set_spec(str(raw_spec))
            except ValueError:
                continue
            for spec in expanded:
                if spec.label in seen:
                    continue
                seen.add(spec.label)
                parsed.append((spec.label, spec.terms))
        return parsed

    def _overlap_metric_defaults(self) -> dict[str, object]:
        defaults: dict[str, object] = {}
        for label, _ in self._parsed_overlap_specs():
            defaults[f"{label}_n_cells"] = 0
            defaults[f"{label}_mean_integrated_intensity"] = None
            defaults[f"{label}_mean_cell_area"] = None
        return defaults

    def _group_matches_overlap_spec(
        self,
        channel_flags: dict[str, int],
        spec_terms: tuple[tuple[str, bool], ...],
    ) -> bool:
        for channel, is_positive in spec_terms:
            flag = int(channel_flags.get(channel.upper(), 0))
            if is_positive and flag != 1:
                return False
            if not is_positive and flag != 0:
                return False
        return True

    def _overlap_region_metrics(
        self,
        overlap_groups,
    ) -> dict[tuple[int, str], dict[str, object]]:
        specs = self._parsed_overlap_specs()
        if not specs or not overlap_groups:
            return {}

        metrics_by_key: dict[tuple[int, str], dict[str, object]] = {}
        for label, spec_terms in specs:
            grouped_matches: dict[tuple[int, str], list] = defaultdict(list)
            for group in overlap_groups:
                if group.region_id == 0:
                    continue
                if not self._group_matches_overlap_spec(group.channel_flags, spec_terms):
                    continue
                grouped_matches[(group.region_id, "total")].append(group)
                if group.hemisphere in {"left", "right"}:
                    grouped_matches[(group.region_id, group.hemisphere)].append(group)
            for key, matched_groups in grouped_matches.items():
                metrics = metrics_by_key.setdefault(key, {})
                metrics[f"{label}_n_cells"] = len(matched_groups)
                metrics[f"{label}_mean_integrated_intensity"] = (
                    float(np.mean([group.mean_intensity for group in matched_groups])) if matched_groups else None
                )
                metrics[f"{label}_mean_cell_area"] = (
                    float(np.mean([group.mean_cell_area for group in matched_groups])) if matched_groups else None
                )
        return metrics_by_key

    def _bundle_output_name(self, bundle: SectionBundle) -> str:
        if bundle.image_channel and bundle.image_channel.lower() not in bundle.channel.lower():
            return f"{bundle.channel}_{bundle.image_channel}"
        return bundle.channel

    def _select_registration_bundle(self, bundles: list[SectionBundle]) -> SectionBundle | None:
        if not bundles:
            return None
        requested = self.config.processing.registration_image_channel.strip().upper()
        if requested:
            for bundle in bundles:
                if bundle.image_channel.upper() == requested:
                    return bundle
        return bundles[0]

    def _registered_section_processing_fingerprint(self) -> dict[str, object]:
        """Return the atlas-rasterization settings that affect section map quality."""

        processing = self.config.processing
        return {
            "midline_threshold_um": float(processing.hemisphere_midline_threshold_um),
            "chunk_rows": int(processing.overlay_chunk_rows),
            "smooth_regions": bool(processing.region_smoothing_enabled),
            "smoothing_kernel_size": int(processing.region_smoothing_kernel_size),
            "smoothing_iterations": int(processing.region_smoothing_iterations),
            "smoothing_downsample_factor": int(processing.region_smoothing_downsample_factor),
            "simplify_contours": bool(processing.region_contour_simplification_enabled),
            "contour_tolerance_px": float(processing.region_contour_simplification_tolerance_px),
            "contour_min_component_area_px": int(processing.region_contour_min_component_area_px),
            "atlas_sampling_mode": str(processing.atlas_sampling_mode),
            "atlas_sampling_radius_vox": int(processing.atlas_sampling_radius_vox),
            "atlas_sampling_batch_size": int(processing.atlas_sampling_batch_size),
        }

    def _resolve_registered_maps(
        self,
        *,
        bundle: SectionBundle,
        atlas: AtlasRepository,
        registration_slice: RegistrationSlice,
        registration_bundle: SectionBundle,
        output_shape: tuple[int, int],
        registered_map_cache: dict[str, RegisteredSectionCacheEntry],
    ) -> tuple[RegisteredSectionCacheEntry, str, float]:
        """Build or load the registered atlas maps for one section image shape."""

        started = time.perf_counter()
        cache_key = build_registered_section_cache_key(
            registration_slice=registration_slice,
            output_shape=output_shape,
            atlas=atlas,
            processing_fingerprint=self._registered_section_processing_fingerprint(),
        )
        if cache_key in registered_map_cache:
            return registered_map_cache[cache_key], "memory", time.perf_counter() - started

        cache_enabled = bool(getattr(self.config.processing, "registered_section_cache_enabled", True))
        cache_path = build_registered_section_cache_path(
            self.config.atlas.registered_section_cache_dir,
            animal_id=bundle.animal_id,
            section_id=bundle.section_id,
            cache_key=cache_key,
        )
        if cache_enabled and cache_path.exists():
            try:
                entry = load_registered_section_cache(cache_path)
                registered_map_cache[cache_key] = entry
                self.logger.info("Loaded section atlas cache: %s", cache_path)
                return entry, "disk", time.perf_counter() - started
            except Exception:
                self.logger.exception("Failed loading cached section atlas map: %s", cache_path)

        region_map, hemisphere_map, qc_metrics = build_registered_maps(
            atlas=atlas,
            registration_slice=registration_slice,
            output_shape=output_shape,
            midline_threshold_um=self.config.processing.hemisphere_midline_threshold_um,
            chunk_rows=self.config.processing.overlay_chunk_rows,
            smooth_regions=self.config.processing.region_smoothing_enabled,
            smoothing_kernel_size=self.config.processing.region_smoothing_kernel_size,
            smoothing_iterations=self.config.processing.region_smoothing_iterations,
            smoothing_downsample_factor=self.config.processing.region_smoothing_downsample_factor,
            simplify_contours=self.config.processing.region_contour_simplification_enabled,
            contour_tolerance_px=self.config.processing.region_contour_simplification_tolerance_px,
            contour_min_component_area_px=self.config.processing.region_contour_min_component_area_px,
            atlas_sampling_mode=self.config.processing.atlas_sampling_mode,
            atlas_sampling_radius_vox=self.config.processing.atlas_sampling_radius_vox,
            atlas_sampling_batch_size=self.config.processing.atlas_sampling_batch_size,
        )
        entry = RegisteredSectionCacheEntry(
            region_map=region_map,
            hemisphere_map=hemisphere_map,
            qc_metrics=qc_metrics,
            cache_path=cache_path if cache_enabled else None,
        )
        registered_map_cache[cache_key] = entry

        if cache_enabled:
            try:
                save_registered_section_cache(
                    cache_path,
                    region_map=region_map,
                    hemisphere_map=hemisphere_map,
                    qc_metrics=qc_metrics,
                )
                self.logger.info("Saved section atlas cache: %s", cache_path)
            except Exception:
                self.logger.exception("Failed saving cached section atlas map: %s", cache_path)

        return entry, "computed", time.perf_counter() - started

    def _annotate_object_patch_metadata(
        self,
        bundle: SectionBundle,
        detected_objects: list,
        region_map: np.ndarray,
        hemisphere_map: np.ndarray,
    ) -> None:
        """Assign signed atlas display codes and connected patch IDs to detected objects."""

        if not detected_objects:
            return
        display_code_map = _display_code_map_from_region_maps(region_map, hemisphere_map)
        component_cache: dict[int, np.ndarray] = {}

        def component_labels_for_code(display_code: int) -> np.ndarray:
            labels = component_cache.get(display_code)
            if labels is None:
                mask = display_code_map == int(display_code)
                if not np.any(mask):
                    labels = np.zeros(display_code_map.shape, dtype=np.int32)
                else:
                    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
                    labels = labels.astype(np.int32, copy=False)
                component_cache[display_code] = labels
            return labels

        height, width = display_code_map.shape
        for obj in detected_objects:
            obj.atlas_display_code = 0
            obj.atlas_patch_component = 0
            obj.atlas_patch_id = ""

            if obj.region_id <= 0 or obj.hemisphere not in {"left", "right"}:
                continue
            display_code = int(-obj.region_id if obj.hemisphere == "left" else obj.region_id)
            component_labels = component_labels_for_code(display_code)
            x = int(round(obj.centroid_x_px))
            y = int(round(obj.centroid_y_px))
            if 0 <= x < width and 0 <= y < height:
                component_label = int(component_labels[y, x])
            else:
                component_label = 0
            if component_label <= 0:
                x0, y0 = obj.bbox_origin
                h, w = obj.mask_crop.shape
                x1 = min(x0 + w, width)
                y1 = min(y0 + h, height)
                if x1 > x0 and y1 > y0:
                    code_crop = display_code_map[y0:y1, x0:x1]
                    label_crop = component_labels[y0:y1, x0:x1]
                    mask_crop = obj.mask_crop[: y1 - y0, : x1 - x0]
                    valid_labels = label_crop[np.asarray(mask_crop, dtype=bool) & (code_crop == display_code)]
                    valid_labels = valid_labels[valid_labels > 0]
                    if valid_labels.size:
                        label_counts = np.bincount(valid_labels.astype(np.int32))
                        component_label = int(label_counts.argmax())
            if component_label <= 0:
                continue
            obj.atlas_display_code = display_code
            obj.atlas_patch_component = component_label
            obj.atlas_patch_id = _format_atlas_patch_id(
                bundle.animal_id,
                bundle.section_id,
                display_code,
                component_label,
            )

    def _min_area_for_bundle(self, bundle: SectionBundle) -> int:
        overrides = {key.upper(): int(value) for key, value in self.config.processing.per_channel_min_area_px.items()}
        if bundle.image_channel.upper() in overrides:
            return overrides[bundle.image_channel.upper()]
        if bundle.channel.upper() in overrides:
            return overrides[bundle.channel.upper()]
        return int(self.config.processing.min_component_area_px)

    def _max_area_for_bundle(self, bundle: SectionBundle) -> int:
        overrides = {key.upper(): int(value) for key, value in self.config.processing.per_channel_max_area_px.items()}
        if bundle.image_channel.upper() in overrides:
            return overrides[bundle.image_channel.upper()]
        if bundle.channel.upper() in overrides:
            return overrides[bundle.channel.upper()]
        return int(self.config.processing.max_component_area_px)

    def _normalize_ilastik_masks(self, groups: list[SectionGroup]) -> pd.DataFrame:
        """Normalize ilastik masks to 0/255 using an auto-inferred foreground class."""

        unique_bundles: dict[Path, SectionBundle] = {}
        for group in groups:
            for bundle in group.bundles:
                if not self._bundle_uses_normalizable_ilastik_mask(bundle):
                    continue
                unique_bundles.setdefault(bundle.mask_path.resolve(), bundle)

        report_rows: list[dict[str, object]] = []
        for mask_path, bundle in sorted(unique_bundles.items(), key=lambda item: str(item[0]).lower()):
            report = normalize_ilastik_mask_file_inplace(
                mask_path,
                min_area_px=self._min_area_for_bundle(bundle),
                max_area_px=self._max_area_for_bundle(bundle),
            )
            report.update(
                {
                    "animal_id": bundle.animal_id,
                    "section_id": bundle.section_id,
                    "channel": bundle.channel,
                    "image_channel": bundle.image_channel,
                }
            )
            report_rows.append(report)
            self.logger.info(
                "Normalized ilastik mask %s using foreground value %s",
                mask_path,
                report["chosen_foreground_value"],
            )
        return pd.DataFrame(report_rows)

    @staticmethod
    def _bundle_uses_normalizable_ilastik_mask(bundle: SectionBundle) -> bool:
        """Return True when a bundle should participate in ilastik mask normalization."""

        mask_source = str(bundle.mask_source).strip().lower()
        if mask_source == "ilastik":
            return True
        if mask_source == "raw":
            lowered_name = bundle.mask_path.name.lower()
            return "simple segmentation" in lowered_name or "simple_segmentation" in lowered_name
        return False

    def _apply_watershed_for_bundle(self, bundle: SectionBundle) -> bool:
        overrides = {
            key.upper(): bool(value) for key, value in getattr(self.config.processing, "per_channel_apply_watershed", {}).items()
        }
        if bundle.image_channel.upper() in overrides:
            return overrides[bundle.image_channel.upper()]
        if bundle.channel.upper() in overrides:
            return overrides[bundle.channel.upper()]
        return bool(self.config.processing.apply_watershed_to_masks)

    def _watershed_marker_threshold_for_bundle(self, bundle: SectionBundle) -> float | str:
        overrides = {
            key.upper(): self._normalize_watershed_marker_threshold_value(value)
            for key, value in getattr(self.config.processing, "per_channel_watershed_marker_threshold_px", {}).items()
        }
        if bundle.image_channel.upper() in overrides:
            return overrides[bundle.image_channel.upper()]
        if bundle.channel.upper() in overrides:
            return overrides[bundle.channel.upper()]
        return self._normalize_watershed_marker_threshold_value(self.config.processing.watershed_marker_threshold_px)

    @staticmethod
    def _normalize_watershed_marker_threshold_value(value: object) -> float | str:
        stripped = str(value).strip().lower()
        if stripped in {"", "auto"}:
            return "auto"
        return max(0.5, float(stripped))

    def _watershed_selective_area_percentile_for_bundle(self, bundle: SectionBundle) -> float:
        overrides = {
            key.upper(): float(value)
            for key, value in getattr(self.config.processing, "per_channel_watershed_selective_area_percentile", {}).items()
        }
        if bundle.image_channel.upper() in overrides:
            value = overrides[bundle.image_channel.upper()]
        elif bundle.channel.upper() in overrides:
            value = overrides[bundle.channel.upper()]
        else:
            value = float(getattr(self.config.processing, "watershed_selective_area_percentile", 90.0) or 90.0)
        return float(np.clip(value, 0.0, 100.0))

    def _watershed_selective_elongation_threshold_for_bundle(self, bundle: SectionBundle) -> float:
        overrides = {
            key.upper(): float(value)
            for key, value in getattr(self.config.processing, "per_channel_watershed_selective_elongation_threshold", {}).items()
        }
        if bundle.image_channel.upper() in overrides:
            value = overrides[bundle.image_channel.upper()]
        elif bundle.channel.upper() in overrides:
            value = overrides[bundle.channel.upper()]
        else:
            value = float(getattr(self.config.processing, "watershed_selective_elongation_threshold", 2.0) or 2.0)
        return max(1.0, value)

    def _mask_threshold_for_bundle(self, bundle: SectionBundle) -> int:
        overrides = {key.upper(): int(value) for key, value in self.config.processing.per_channel_mask_threshold.items()}
        if bundle.image_channel.upper() in overrides:
            return overrides[bundle.image_channel.upper()]
        if bundle.channel.upper() in overrides:
            return overrides[bundle.channel.upper()]
        return int(self.config.processing.mask_threshold)

    def _save_group_overlay(
        self,
        group_results: list[SectionChannelResult],
        atlas: AtlasRepository,
        output_dir: Path,
    ) -> dict[str, object]:
        full_dir = output_dir / "overlay"
        animal_id = group_results[0].bundle.animal_id
        section_id = group_results[0].bundle.section_id
        full_path = full_dir / f"{animal_id}_{section_id}_overlay_full.tiff"
        channel_images = [
            (
                result.bundle.image_channel or result.bundle.channel,
                np.asarray(result.source_image)
                if result.source_image is not None
                else load_image_array(result.bundle.image_path, grayscale=False),
            )
            for result in group_results
        ]
        overlay_assets = save_multichannel_overlay_images(
            channel_images=channel_images,
            section_results=group_results,
            atlas=atlas,
            full_out=full_path,
            full_max_size=self.config.processing.overlay_full_max_size,
            draw_masks=self.config.processing.combined_overlay_draw_masks,
            draw_centroids=self.config.processing.combined_overlay_draw_centroids,
            channel_colors=self.config.processing.overlay_channel_colors,
            png_compress_level=self.config.processing.overlay_png_compress_level,
            tiff_compression=self.config.processing.overlay_tiff_compression,
        )
        return {
            "combined_overlay_full": overlay_assets["full"],
            "channel_map_frame": overlay_assets["channel_map_frame"],
        }

    def _attach_combined_overlay_paths(
        self,
        group_results: list[SectionChannelResult],
        asset_paths: dict[str, object],
    ) -> None:
        for result in group_results:
            result.combined_overlay_preview_path = None
            result.combined_overlay_full_path = asset_paths.get("combined_overlay_full")
            result.numbered_atlas_preview_path = None
            result.numbered_atlas_full_path = asset_paths.get("combined_overlay_full")
            result.overlay_full_path = asset_paths.get("combined_overlay_full")
            result.roi_legend_path = None
            qc_metrics = json.loads(str(result.section_summary.get("registration_qc_metrics", "{}")))
            for key, value in asset_paths.items():
                if isinstance(value, Path):
                    qc_metrics[key] = str(value)
            result.section_summary["registration_qc_metrics"] = metrics_json(qc_metrics)

    def _write_animal_channel_map_workbooks(
        self,
        output_dir: Path,
        channel_maps: dict[str, pd.DataFrame],
    ) -> dict[str, Path]:
        workbook_dir = output_dir / "overlay"
        workbook_dir.mkdir(parents=True, exist_ok=True)
        workbook_paths: dict[str, Path] = {}
        for animal_id in sorted(channel_maps):
            workbook_path = workbook_dir / f"{animal_id}_multichannel_channel_maps.xlsx"
            fallback_csv_path = workbook_dir / f"{animal_id}_multichannel_channel_maps.csv"
            legacy_workbook_path = output_dir / "channel_maps" / f"{animal_id}_multichannel_channel_maps.xlsx"
            if legacy_workbook_path.exists():
                legacy_workbook_path.unlink()
            try:
                with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
                    channel_maps[animal_id].to_excel(writer, sheet_name="overlay_stack", index=False)
                if fallback_csv_path.exists():
                    fallback_csv_path.unlink()
                workbook_paths[animal_id] = workbook_path
            except Exception as exc:
                self.logger.warning(
                    "Failed writing channel-map workbook for %s (%s); falling back to CSV.",
                    animal_id,
                    exc,
                )
                if workbook_path.exists():
                    workbook_path.unlink()
                channel_maps[animal_id].to_csv(fallback_csv_path, index=False)
                workbook_paths[animal_id] = fallback_csv_path
        return workbook_paths

    def _attach_animal_channel_map_paths(
        self,
        results: list[SectionChannelResult],
        workbook_paths: dict[str, Path],
    ) -> None:
        for result in results:
            workbook_path = workbook_paths.get(result.bundle.animal_id)
            if workbook_path is None:
                continue
            qc_metrics = json.loads(str(result.section_summary.get("registration_qc_metrics", "{}")))
            qc_metrics["animal_multichannel_channel_map_path"] = str(workbook_path)
            qc_metrics["animal_multichannel_channel_map_format"] = workbook_path.suffix.lower().lstrip(".")
            if workbook_path.suffix.lower() == ".xlsx":
                qc_metrics["animal_multichannel_channel_map_sheet"] = "overlay_stack"
            else:
                qc_metrics.pop("animal_multichannel_channel_map_sheet", None)
            result.section_summary["registration_qc_metrics"] = metrics_json(qc_metrics)

    def _cleanup_obsolete_outputs(
        self,
        output_dir: Path,
        current_tables: set[str],
        comparison_enabled: bool,
    ) -> None:
        """Remove legacy summary exports that should not survive native-only runs."""

        obsolete_table_names = {
            "section_summary_native.csv",
            "section_summary_reference.csv",
            "region_summary_native.csv",
            "region_summary_reference.csv",
            "cell_level_reference.csv",
            "comparison_report.csv",
            "multichannel_summary.csv",
            "boundary_metrics_vs_25um.csv",
            "highres10_vs_25um_metrics.csv",
            "overlay_full_shrink_report.csv",
            "unassigned_qc_summary.csv",
        }
        for filename in obsolete_table_names - current_tables:
            path = output_dir / filename
            if path.exists():
                path.unlink()
        if not comparison_enabled:
            comparison_md = output_dir / "comparison_report.md"
            if comparison_md.exists():
                comparison_md.unlink()

        legacy_overlay_dir = output_dir / "overlays" / "full"
        if legacy_overlay_dir.exists():
            for pattern in (
                "*_overlay_full.png",
                "*_overlay_full.tif",
                "*_overlay_full.tiff",
                "*_multichannel_overlay_full.png",
                "*_multichannel_overlay_full.tif",
                "*_multichannel_atlas_full.tif",
                "*_multichannel_atlas_numbered_full.tif",
                "*_atlas_numbered_full.tif",
                "*_atlas_numbered_full.png",
                "*_channel_map.csv",
                "*_multichannel_channel_maps.xlsx",
                "*_multichannel_channel_maps.csv",
            ):
                for stale_path in legacy_overlay_dir.glob(pattern):
                    if stale_path.is_file():
                        stale_path.unlink()

        current_overlay_dir = output_dir / "overlay"
        if current_overlay_dir.exists():
            for pattern in (
                "*_overlay_full.png",
                "*_multichannel_overlay_full.png",
                "*_multichannel_overlay_full.tif",
                "*_multichannel_atlas_full.tif",
                "*_multichannel_atlas_numbered_full.tif",
                "*_atlas_numbered_full.tif",
                "*_atlas_numbered_full.png",
                "*_channel_map.csv",
                "*_multichannel_channel_maps.csv",
            ):
                for stale_path in current_overlay_dir.glob(pattern):
                    if stale_path.is_file():
                        stale_path.unlink()

        preview_dir = output_dir / "overlays" / "preview"
        if preview_dir.exists():
            for stale_path in preview_dir.glob("*"):
                if stale_path.is_file():
                    stale_path.unlink()
            if not any(preview_dir.iterdir()):
                preview_dir.rmdir()

        roi_legend_dir = output_dir / "overlays" / "roi_legend"
        if roi_legend_dir.exists():
            for stale_path in roi_legend_dir.glob("*"):
                if stale_path.is_file():
                    stale_path.unlink()
            if not any(roi_legend_dir.iterdir()):
                roi_legend_dir.rmdir()

        overlays_dir = output_dir / "overlays"
        if overlays_dir.exists():
            for child in overlays_dir.iterdir():
                if child.is_dir() and not any(child.iterdir()):
                    child.rmdir()
            if not any(overlays_dir.iterdir()):
                overlays_dir.rmdir()
