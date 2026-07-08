"""Configuration models and file IO helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AtlasConfig:
    """Atlas-related settings."""

    name: str = "allen_mouse_ccfv3_2017_10um"
    labels_path: Path = Path("../../atlas_cache/allen_ccf/annotation_10.nrrd")
    tree_path: Path = Path("../../QCAlign-v0.8/ABA_Mouse_CCFv3_2017_25um.cutlas/tree.json")
    labels_txt_path: Path = Path("../../QCAlign-v0.8/ABA_Mouse_CCFv3_2017_25um.cutlas/labels.txt")
    registered_section_cache_dir: Path = Path("./atlas_cache/registered_sections")
    voxel_size_um: float = 10.0
    quicknii_resolution_vox: tuple[int, int, int] = (1140, 1320, 800)
    allen_bregma_um: tuple[float, float, float] = (5400.0, 0.0, 5700.0)
    stereotactic_tilt_deg: float = 0.0
    dv_scale: float = 1.0


@dataclass(slots=True)
class DiscoveryConfig:
    """Folder discovery rules."""

    project_root: Path = Path("")
    image_folder: Path = Path("")
    segmentation_folder: Path = Path("")
    atlas_json_path: Path = Path("")
    animal_include: list[str] = field(default_factory=lambda: ["505A"])
    channel_include: list[str] = field(default_factory=lambda: ["cfos"])
    section_include: list[str] = field(default_factory=lambda: ["XY01"])
    image_subdir_name: str = "png"
    mask_subdir_candidates: list[str] = field(
        default_factory=lambda: ["ilastik", "ilastik/glansbayPNGconverted", "mask", "masks"]
    )
    mask_filename_globs: list[str] = field(
        default_factory=lambda: [
            "*Simple Segmentation*.png",
            "*Simple_Segmentation*.png",
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.tif",
            "*.tiff",
        ]
    )
    registration_subdir_candidates: list[str] = field(default_factory=lambda: ["jpg"])
    registration_filename_globs: list[str] = field(
        default_factory=lambda: ["*VA*.json", "*QuickNII*.json", "*.json"]
    )
    existing_result_subdir_candidates: list[str] = field(default_factory=lambda: ["result", "results"])
    recursive_channel_search: bool = False


@dataclass(slots=True)
class ProcessingConfig:
    """Single-channel quantification settings."""

    mask_threshold: int = 1
    normalize_ilastik_masks_to_binary: bool = False
    min_component_area_px: int = 9
    max_component_area_px: int = 0
    analysis_image_channels: list[str] = field(default_factory=list)
    registration_image_channel: str = ""
    per_channel_min_area_px: dict[str, int] = field(default_factory=dict)
    per_channel_max_area_px: dict[str, int] = field(default_factory=dict)
    per_channel_mask_threshold: dict[str, int] = field(default_factory=dict)
    per_channel_apply_watershed: dict[str, bool] = field(default_factory=dict)
    per_channel_watershed_marker_threshold_px: dict[str, float | str] = field(default_factory=dict)
    per_channel_watershed_selective_area_percentile: dict[str, float] = field(default_factory=dict)
    per_channel_watershed_selective_elongation_threshold: dict[str, float] = field(default_factory=dict)
    watershed_selective_area_percentile: float = 90.0
    watershed_selective_elongation_threshold: float = 2.0
    border_assignment_policy: str = "bigger"
    apply_watershed_to_masks: bool = False
    watershed_marker_threshold_px: float | str = 1.5
    hemisphere_midline_threshold_um: float = 75.0
    coordinate_unit: str = "mm"
    region_smoothing_enabled: bool = True
    region_smoothing_kernel_size: int = 7
    region_smoothing_iterations: int = 2
    region_smoothing_downsample_factor: int = 3
    region_contour_simplification_enabled: bool = False
    region_contour_simplification_tolerance_px: float = 2.5
    region_contour_min_component_area_px: int = 128
    atlas_sampling_mode: str = "nearest"
    atlas_sampling_radius_vox: int = 2
    atlas_sampling_batch_size: int = 4096
    include_region_fill_in_preview: bool = False
    overlay_enabled: bool = True
    overlay_preview_max_size: int = 1400
    overlay_full_max_size: int = 3200
    overlay_png_compress_level: int = 9
    overlay_tiff_compression: str = "tiff_adobe_deflate"
    overlay_draw_masks: bool = False
    overlay_draw_centroids: bool = True
    combined_overlay_enabled: bool = True
    combined_overlay_draw_masks: bool = True
    combined_overlay_draw_centroids: bool = True
    overlay_channel_colors: dict[str, list[int]] = field(
        default_factory=lambda: {
            "CH1": [90, 220, 255],
            "CH2": [255, 110, 110],
            "CH3": [130, 255, 150],
            "CH4": [255, 195, 90],
        }
    )
    overlay_chunk_rows: int = 256
    parallel_workers: int = 0
    registered_section_cache_enabled: bool = True
    export_patch_ids: bool = False


@dataclass(slots=True)
class MatchingConfig:
    """Multichannel matching settings."""

    enabled: bool = True
    method: str = "centroid_distance"
    distance_threshold_px: float = 25.0
    iou_threshold: float = 0.15
    overlap_threshold_px: int = 25
    combinations: list[str] = field(default_factory=list)
    pair_rules: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class ComparisonConfig:
    """Existing Nutil result comparison settings."""

    enabled: bool = False
    max_mean_nn_distance_um_warning: float = 150.0
    max_abs_count_delta_warning: int = 20


@dataclass(slots=True)
class ReferenceCompatibilityConfig:
    """Reference-assisted export settings when existing Nutil outputs are present."""

    enabled: bool = False
    prefer_existing_overlay_if_available: bool = False
    prefer_reference_summary_if_available: bool = False
    export_reference_cell_level: bool = False
    overlay_alpha: float = 0.28
    mode_filter_size: int = 5
    roi_number_min_area_px: int = 20000


@dataclass(slots=True)
class OutputConfig:
    """Output folder settings."""

    output_root: Path = Path("./outputs")
    run_name: str = "single_section_demo"
    save_resolved_config: bool = True

    @property
    def output_dir(self) -> Path:
        """Concrete run directory."""

        return self.output_root / self.run_name


@dataclass(slots=True)
class AppConfig:
    """Top-level app configuration."""

    atlas: AtlasConfig = field(default_factory=AtlasConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    reference: ReferenceCompatibilityConfig = field(default_factory=ReferenceCompatibilityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def resolve_paths(self, config_path: Path | None = None) -> "AppConfig":
        """Resolve relative filesystem paths against the config file location."""

        base_dir = config_path.parent if config_path else Path.cwd()
        self.atlas.labels_path = _resolve_atlas_labels_path(self.atlas, base_dir)
        self.atlas.tree_path = _resolve_atlas_tree_path(self.atlas, base_dir)
        self.atlas.labels_txt_path = _resolve_atlas_labels_txt_path(self.atlas, base_dir)
        self.atlas.registered_section_cache_dir = _resolve_registered_section_cache_dir(
            Path(self.atlas.registered_section_cache_dir),
            base_dir,
        )
        self.discovery.project_root = _resolve_path_value(Path(self.discovery.project_root), base_dir)
        self.discovery.image_folder = _resolve_optional_path(Path(self.discovery.image_folder), base_dir)
        self.discovery.segmentation_folder = _resolve_optional_path(Path(self.discovery.segmentation_folder), base_dir)
        self.discovery.atlas_json_path = _resolve_optional_path(Path(self.discovery.atlas_json_path), base_dir)
        self.output.output_root = (base_dir / self.output.output_root).resolve()
        return self

    def to_serializable_dict(self) -> dict[str, Any]:
        """Convert the config into a YAML-safe dictionary."""

        data = asdict(self)
        for group_name in ("atlas", "discovery", "output"):
            for key, value in list(data[group_name].items()):
                if isinstance(value, Path) or "path" in key or "root" in key or "folder" in key:
                    if (
                        group_name == "discovery"
                        and key in {"image_folder", "segmentation_folder", "atlas_json_path"}
                        and Path(value) == Path(".")
                    ):
                        data[group_name][key] = ""
                    else:
                        data[group_name][key] = str(value)
        data.get("atlas", {}).pop("registered_section_cache_dir", None)
        return data


def _merge_mapping(defaults: Any, raw: dict[str, Any]) -> Any:
    """Recursively merge plain dictionaries into dataclasses."""

    for field_name in defaults.__dataclass_fields__:
        current_value = getattr(defaults, field_name)
        if field_name not in raw:
            continue
        incoming = raw[field_name]
        if hasattr(current_value, "__dataclass_fields__"):
            _merge_mapping(current_value, incoming)
        else:
            setattr(defaults, field_name, incoming)
    return defaults


def load_app_config(path: Path) -> AppConfig:
    """Load a YAML or JSON-like config file into an AppConfig."""

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    _upgrade_legacy_watershed_config(raw)
    config = _merge_mapping(AppConfig(), raw)
    return config.resolve_paths(path)


def _upgrade_legacy_watershed_config(raw: dict[str, Any]) -> None:
    """Translate old watershed min-distance keys into marker-threshold keys."""

    processing = raw.get("processing")
    if not isinstance(processing, dict):
        return
    if (
        "per_channel_watershed_marker_threshold_px" not in processing
        and "per_channel_watershed_min_distance_px" in processing
    ):
        processing["per_channel_watershed_marker_threshold_px"] = processing["per_channel_watershed_min_distance_px"]
    if "watershed_marker_threshold_px" not in processing and "watershed_min_distance_px" in processing:
        processing["watershed_marker_threshold_px"] = processing["watershed_min_distance_px"]


def _resolve_existing_path(path_value: Path, base_dir: Path) -> Path:
    """Resolve a path and try a few workspace-aware fallbacks when it moved."""

    candidate = (base_dir / path_value).resolve()
    if candidate.exists():
        return candidate

    normalized_parts = [part for part in path_value.parts if part not in {"..", "."}]
    suffixes: list[Path] = []
    if normalized_parts:
        suffixes.append(Path(*normalized_parts))
        if normalized_parts[0] != "QUINTsoftware":
            suffixes.append(Path("QUINTsoftware") / Path(*normalized_parts))

    for root in _search_roots(base_dir):
        for suffix in suffixes:
            alt = (root / suffix).resolve()
            if alt.exists():
                return alt
    return candidate


def _resolve_path_value(path_value: Path, base_dir: Path) -> Path:
    """Resolve a path-like value against the config file directory."""

    return (base_dir / path_value).resolve()


def _resolve_optional_path(path_value: Path, base_dir: Path) -> Path:
    """Resolve an optional path, preserving an empty value when unset."""

    raw_value = str(path_value).strip()
    if raw_value in {"", "."}:
        return Path("")
    return _resolve_existing_path(path_value, base_dir)


def _resolve_registered_section_cache_dir(path_value: Path, base_dir: Path) -> Path:
    """Resolve registered-section cache dir with portable fallback for stale absolute paths."""

    raw_value = str(path_value).strip()
    default_relative = Path("./atlas_cache/registered_sections")
    bundle_root_text = os.environ.get("QUINT_PORTABLE_BUNDLE_ROOT", "").strip()
    shared_portable_default = None
    if bundle_root_text:
        try:
            bundle_root = Path(bundle_root_text).resolve()
            shared_portable_default = bundle_root.parent / "QDF_shared_cache" / "registered_sections"
        except Exception:
            shared_portable_default = None

    def _portable_or_local_default() -> Path:
        if shared_portable_default is not None:
            return shared_portable_default
        return (base_dir / default_relative).resolve()

    if raw_value in {"", "."}:
        return _portable_or_local_default()

    if not path_value.is_absolute():
        return (base_dir / path_value).resolve()

    try:
        resolved = path_value.resolve()
    except Exception:
        resolved = path_value
    if resolved.exists():
        return resolved

    return _portable_or_local_default()


def _resolve_atlas_labels_path(atlas: AtlasConfig, base_dir: Path) -> Path:
    """Resolve atlas labels from explicit config first, then from local cache locations."""

    explicit = _resolve_existing_path(Path(atlas.labels_path), base_dir)
    if explicit.exists():
        return explicit

    candidates: list[Path] = []
    requested_name = Path(atlas.labels_path).name.lower()
    atlas_name = atlas.name.lower()
    requested_text = str(atlas.labels_path).lower()
    is_10um = abs(float(atlas.voxel_size_um) - 10.0) < 0.6 or "10um" in atlas_name or "annotation_10" in requested_text
    is_25um = abs(float(atlas.voxel_size_um) - 25.0) < 0.6 or "25um" in atlas_name or "annotation_25" in requested_text

    if is_10um:
        candidates.extend(
            [
                Path("atlas") / "ccf" / "annotation_10.nrrd",
                Path("atlas_cache") / "allen_ccf" / "annotation_10.nrrd",
            ]
        )
    if is_25um:
        candidates.extend(
            [
                Path("atlas") / "ccf" / "annotation_25.nrrd",
                Path("atlas_cache") / "allen_ccf" / "annotation_25.nrrd",
                Path("QUINTsoftware") / "QCAlign-v0.8" / "ABA_Mouse_CCFv3_2017_25um.cutlas" / "labels.nii.gz",
            ]
        )
    if requested_name == "labels.nii.gz":
        candidates.append(Path("QUINTsoftware") / "QCAlign-v0.8" / "ABA_Mouse_CCFv3_2017_25um.cutlas" / "labels.nii.gz")

    seen: set[Path] = set()
    for root in _search_roots(base_dir):
        for suffix in candidates:
            alt = (root / suffix).resolve()
            if alt in seen:
                continue
            seen.add(alt)
            if alt.exists():
                return alt
    return explicit


def _resolve_atlas_tree_path(atlas: AtlasConfig, base_dir: Path) -> Path:
    """Resolve atlas tree metadata from explicit config or portable cache locations."""

    explicit = _resolve_existing_path(Path(atlas.tree_path), base_dir)
    if explicit.exists():
        return explicit
    candidates = [
        Path("atlas") / "ccf" / "tree.json",
        Path("atlas_cache") / "allen_ccf" / "tree.json",
        Path("QUINTsoftware") / "QCAlign-v0.8" / "ABA_Mouse_CCFv3_2017_25um.cutlas" / "tree.json",
    ]
    for root in _search_roots(base_dir):
        for suffix in candidates:
            alt = (root / suffix).resolve()
            if alt.exists():
                return alt
    return explicit


def _resolve_atlas_labels_txt_path(atlas: AtlasConfig, base_dir: Path) -> Path:
    """Resolve atlas label text metadata from explicit config or portable cache locations."""

    explicit = _resolve_existing_path(Path(atlas.labels_txt_path), base_dir)
    if explicit.exists():
        return explicit
    candidates = [
        Path("atlas") / "ccf" / "labels.txt",
        Path("atlas_cache") / "allen_ccf" / "labels.txt",
        Path("QUINTsoftware") / "QCAlign-v0.8" / "ABA_Mouse_CCFv3_2017_25um.cutlas" / "labels.txt",
    ]
    for root in _search_roots(base_dir):
        for suffix in candidates:
            alt = (root / suffix).resolve()
            if alt.exists():
                return alt
    return explicit


def _search_roots(base_dir: Path) -> list[Path]:
    """Return ordered root directories to search for moved resources."""

    roots: list[Path] = [base_dir, *base_dir.parents]
    cwd = Path.cwd().resolve()
    bundle_root_text = os.environ.get("QUINT_PORTABLE_BUNDLE_ROOT", "").strip()
    extras = [cwd]
    if bundle_root_text:
        extras.append(Path(bundle_root_text).resolve())
    runtime_root_text = os.environ.get("QDF_RUNTIME_ROOT", "").strip()
    if runtime_root_text:
        extras.append(Path(runtime_root_text).resolve())
    extras.append(Path.home() / "Documents" / "QDF_portable")
    seen: set[Path] = set()
    ordered: list[Path] = []
    for root in roots + extras:
        try:
            resolved = root.resolve()
        except Exception:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return ordered


def save_app_config(config: AppConfig, path: Path) -> None:
    """Save the resolved app config to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = config.to_serializable_dict()
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def gui_visible_config_dict(config: AppConfig) -> dict[str, Any]:
    """Return the subset of config values that are directly represented in QUINTdeepflow2 GUI."""

    return {
        "discovery": {
            "image_folder": str(config.discovery.image_folder),
            "segmentation_folder": str(config.discovery.segmentation_folder),
            "atlas_json_path": str(config.discovery.atlas_json_path),
        },
        "processing": {
            "analysis_image_channels": list(config.processing.analysis_image_channels),
            "normalize_ilastik_masks_to_binary": bool(config.processing.normalize_ilastik_masks_to_binary),
            "per_channel_min_area_px": dict(config.processing.per_channel_min_area_px),
            "per_channel_max_area_px": dict(config.processing.per_channel_max_area_px),
            "per_channel_apply_watershed": dict(config.processing.per_channel_apply_watershed),
            "per_channel_watershed_marker_threshold_px": dict(config.processing.per_channel_watershed_marker_threshold_px),
            "per_channel_watershed_selective_area_percentile": dict(
                config.processing.per_channel_watershed_selective_area_percentile
            ),
            "per_channel_watershed_selective_elongation_threshold": dict(
                config.processing.per_channel_watershed_selective_elongation_threshold
            ),
            "watershed_selective_area_percentile": float(config.processing.watershed_selective_area_percentile),
            "watershed_selective_elongation_threshold": float(config.processing.watershed_selective_elongation_threshold),
            "border_assignment_policy": str(config.processing.border_assignment_policy),
            "parallel_workers": int(config.processing.parallel_workers),
        },
        "matching": {
            "combinations": list(config.matching.combinations),
            "pair_rules": dict(config.matching.pair_rules),
        },
        "output": {
            "output_root": str(config.output.output_root),
            "run_name": str(config.output.run_name),
        },
    }


def save_gui_visible_app_config(config: AppConfig, path: Path) -> None:
    """Save only GUI-visible QUINTdeepflow2 settings."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(gui_visible_config_dict(config), handle, sort_keys=False, allow_unicode=True)
