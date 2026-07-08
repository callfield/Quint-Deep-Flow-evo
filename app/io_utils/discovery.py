"""Discovery helpers for flexible animal / channel / section folder structures."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from config.settings import AppConfig
from data_models.models import SectionBundle, SectionGroup
from registration.parser import parse_registration_file

LOG = logging.getLogger(__name__)

SECTION_PATTERN = re.compile(r"(xy\d+)", re.IGNORECASE)
IMAGE_CHANNEL_PATTERN = re.compile(r"(ch\d+)", re.IGNORECASE)
MASK_SUFFIXES = (
    "_simple_segmentation",
    " simple segmentation",
    "_simple segmentation",
    "_segmentation",
    "_mask",
)


def _is_explicit_path(path: Path) -> bool:
    raw = str(path).strip()
    return raw not in {"", "."}


def normalize_stem(value: str) -> str:
    """Normalize image / mask / registration names for matching."""

    normalized = Path(value).stem.lower().replace(" ", "_").replace("-", "_")
    for suffix in MASK_SUFFIXES:
        normalized = normalized.replace(suffix, "")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def extract_section_id(value: str) -> str:
    """Extract a stable section id such as XY01 from a filename."""

    match = SECTION_PATTERN.search(value)
    return match.group(1).upper() if match else Path(value).stem


def extract_animal_id(value: str) -> str:
    """Extract an animal/sample prefix before the section token when possible."""

    match = SECTION_PATTERN.search(value)
    if not match:
        return Path(value).stem
    prefix = value[: match.start()]
    prefix = prefix.rstrip("_- ").strip()
    return prefix or Path(value).stem


def extract_image_channel(value: str) -> str:
    """Extract a stable acquisition channel label such as CH2 from a filename."""

    match = IMAGE_CHANNEL_PATTERN.search(value)
    return match.group(1).upper() if match else ""


def _looks_like_segmentation(path: Path) -> bool:
    lowered = path.name.lower()
    return any(suffix in lowered for suffix in MASK_SUFFIXES)


def _collect_files(
    directory: Path,
    patterns: tuple[str, ...],
    *,
    exclude_segmentation_like: bool = False,
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for pattern in patterns:
        for path in sorted(directory.glob(pattern)):
            if path.is_file():
                if exclude_segmentation_like and _looks_like_segmentation(path):
                    continue
                files[normalize_stem(path.name)] = path
    return files


def _mask_filename_priority(name: str) -> int:
    lowered = name.lower()
    if "simple segmentation" in lowered or "simple_segmentation" in lowered:
        return 0
    if "segmentation" in lowered:
        return 1
    if "mask" in lowered:
        return 2
    return 3


def _discover_masks(channel_dir: Path, config: AppConfig) -> dict[str, tuple[Path, str]]:
    """Find masks across configured sources and prefer Simple Segmentation outputs."""

    selected: dict[str, tuple[tuple[int, int, str], Path, str]] = {}
    for source_rank, source_name in enumerate(config.discovery.mask_subdir_candidates):
        mask_dir = channel_dir / source_name
        if not mask_dir.exists():
            continue
        candidates: dict[Path, Path] = {}
        for glob_pattern in config.discovery.mask_filename_globs:
            for path in sorted(mask_dir.glob(glob_pattern)):
                if path.is_file():
                    candidates[path.resolve()] = path
        for path in candidates.values():
            key = normalize_stem(path.name)
            score = (source_rank, _mask_filename_priority(path.name), path.name.lower())
            current = selected.get(key)
            if current is None or score < current[0]:
                selected[key] = (score, path, source_name)
    return {key: (path, source_name) for key, (_, path, source_name) in selected.items()}


def _discover_registration_map(channel_dir: Path, config: AppConfig) -> dict[str, tuple[Path, str]]:
    mapping: dict[str, tuple[Path, str]] = {}
    for subdir_name in config.discovery.registration_subdir_candidates:
        reg_dir = channel_dir / subdir_name
        if not reg_dir.exists():
            continue
        for glob_pattern in config.discovery.registration_filename_globs:
            for json_path in sorted(reg_dir.glob(glob_pattern)):
                try:
                    registration = parse_registration_file(json_path)
                except Exception as exc:  # pragma: no cover
                    LOG.warning("Skipping unreadable registration file %s: %s", json_path, exc)
                    continue
                for reg_slice in registration.slices:
                    mapping[normalize_stem(reg_slice.filename)] = (json_path, reg_slice.filename)
            if mapping:
                return mapping
    return mapping


def _discover_explicit_section_groups(config: AppConfig) -> list[SectionGroup]:
    """Discover bundles from explicitly chosen image / segmentation folders and one atlas JSON."""

    image_dir = config.discovery.image_folder
    segmentation_dir = config.discovery.segmentation_folder
    atlas_json_path = config.discovery.atlas_json_path
    if not image_dir.exists():
        raise FileNotFoundError(f"Image folder does not exist: {image_dir}")
    if not segmentation_dir.exists():
        raise FileNotFoundError(f"Segmentation folder does not exist: {segmentation_dir}")
    if not atlas_json_path.exists():
        raise FileNotFoundError(f"Atlas JSON does not exist: {atlas_json_path}")

    images = _collect_files(
        image_dir,
        ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"),
        exclude_segmentation_like=True,
    )
    if not images:
        raise FileNotFoundError(f"No input images found in: {image_dir}")

    selected: dict[str, tuple[tuple[int, str], Path]] = {}
    candidates: dict[Path, Path] = {}
    for glob_pattern in config.discovery.mask_filename_globs:
        for path in sorted(segmentation_dir.glob(glob_pattern)):
            if path.is_file():
                candidates[path.resolve()] = path
    for path in candidates.values():
        key = normalize_stem(path.name)
        score = (_mask_filename_priority(path.name), path.name.lower())
        current = selected.get(key)
        if current is None or score < current[0]:
            selected[key] = (score, path)
    masks = {key: path for key, (_, path) in selected.items()}
    if not masks:
        raise FileNotFoundError(f"No segmentation masks found in: {segmentation_dir}")

    registration = parse_registration_file(atlas_json_path)
    reg_map = {normalize_stem(reg_slice.filename): reg_slice.filename for reg_slice in registration.slices}
    if not reg_map:
        raise ValueError(f"No registration entries found in atlas JSON: {atlas_json_path}")
    registration_channel = str(config.processing.registration_image_channel or "").upper()
    reg_by_section: dict[str, str] = {}
    for reg_slice in registration.slices:
        section_id = extract_section_id(reg_slice.filename)
        slice_channel = extract_image_channel(reg_slice.filename).upper()
        current = reg_by_section.get(section_id)
        if current is None:
            reg_by_section[section_id] = reg_slice.filename
            continue
        if registration_channel and slice_channel == registration_channel:
            reg_by_section[section_id] = reg_slice.filename

    grouped: dict[tuple[str, str], list[SectionBundle]] = defaultdict(list)
    analysis_channels = {value.upper() for value in config.processing.analysis_image_channels}
    mask_source = segmentation_dir.name or "segmentation"

    common_keys = sorted(set(images) & set(masks))
    for key in common_keys:
        image_path = images[key]
        animal_id = extract_animal_id(image_path.stem) or image_dir.parent.name or image_dir.name
        image_channel = extract_image_channel(image_path.name)
        if analysis_channels and image_channel not in analysis_channels:
            continue
        section_id = extract_section_id(image_path.name)
        if config.discovery.section_include and section_id not in config.discovery.section_include:
            continue
        registration_filename = reg_map.get(key) or reg_by_section.get(section_id)
        if not registration_filename:
            continue
        grouped[(animal_id, section_id)].append(
            SectionBundle(
                animal_id=animal_id,
                channel=image_channel or "UNKNOWN",
                section_id=section_id,
                image_path=image_path,
                mask_path=masks[key],
                registration_json_path=atlas_json_path,
                registration_filename=registration_filename,
                existing_result_dir=None,
                image_channel=image_channel,
                mask_source=mask_source,
            )
        )

    return [
        SectionGroup(animal_id=resolved_animal_id, section_id=section_id, bundles=sorted(bundles, key=lambda item: item.channel))
        for (resolved_animal_id, section_id), bundles in sorted(grouped.items())
    ]


def discover_section_groups(config: AppConfig) -> list[SectionGroup]:
    """Discover input bundles grouped by animal / section across channels."""

    if (
        _is_explicit_path(config.discovery.image_folder)
        and _is_explicit_path(config.discovery.segmentation_folder)
        and _is_explicit_path(config.discovery.atlas_json_path)
    ):
        return _discover_explicit_section_groups(config)

    project_root = config.discovery.project_root
    if not project_root.exists():
        raise FileNotFoundError(f"Project root does not exist: {project_root}")

    animal_dirs = [
        path
        for path in sorted(project_root.iterdir())
        if path.is_dir() and (not config.discovery.animal_include or path.name in config.discovery.animal_include)
    ]
    grouped: dict[tuple[str, str], list[SectionBundle]] = defaultdict(list)
    analysis_channels = {value.upper() for value in config.processing.analysis_image_channels}

    for animal_dir in animal_dirs:
        channel_dirs = [
            path
            for path in sorted(animal_dir.iterdir())
            if path.is_dir() and (not config.discovery.channel_include or path.name in config.discovery.channel_include)
        ]
        for channel_dir in channel_dirs:
            image_dir = channel_dir / config.discovery.image_subdir_name
            if not image_dir.exists():
                continue
            images = _collect_files(
                image_dir,
                ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"),
                exclude_segmentation_like=True,
            )
            masks = _discover_masks(channel_dir, config)
            if not masks:
                continue
            reg_map = _discover_registration_map(channel_dir, config)
            existing_result_dir = next(
                (
                    channel_dir / name
                    for name in config.discovery.existing_result_subdir_candidates
                    if (channel_dir / name).exists()
                ),
                None,
            )

            common_keys = sorted(set(images) & set(masks) & set(reg_map))
            for key in common_keys:
                image_path = images[key]
                image_channel = extract_image_channel(image_path.name)
                if analysis_channels and image_channel not in analysis_channels:
                    continue
                section_id = extract_section_id(image_path.name)
                if config.discovery.section_include and section_id not in config.discovery.section_include:
                    continue
                reg_json, reg_filename = reg_map[key]
                mask_path, mask_source = masks[key]
                grouped[(animal_dir.name, section_id)].append(
                    SectionBundle(
                        animal_id=animal_dir.name,
                        channel=channel_dir.name,
                        section_id=section_id,
                        image_path=image_path,
                        mask_path=mask_path,
                        registration_json_path=reg_json,
                        registration_filename=reg_filename,
                        existing_result_dir=existing_result_dir,
                        image_channel=image_channel,
                        mask_source=mask_source,
                    )
                )

    return [
        SectionGroup(animal_id=animal_id, section_id=section_id, bundles=sorted(bundles, key=lambda item: item.channel))
        for (animal_id, section_id), bundles in sorted(grouped.items())
    ]


def discovery_to_dataframe(groups: list[SectionGroup]) -> pd.DataFrame:
    """Flatten discovered groups into a GUI-friendly table."""

    rows: list[dict[str, str]] = []
    for group in groups:
        for bundle in group.bundles:
            rows.append(
                {
                    "animal_id": bundle.animal_id,
                    "section_id": bundle.section_id,
                    "channel": bundle.channel,
                    "image_channel": bundle.image_channel,
                    "image_file": str(bundle.image_path),
                    "mask_file": str(bundle.mask_path),
                    "mask_source": bundle.mask_source,
                    "registration_json": str(bundle.registration_json_path),
                    "registration_entry": bundle.registration_filename,
                    "existing_result_dir": str(bundle.existing_result_dir) if bundle.existing_result_dir else "",
                }
            )
    return pd.DataFrame(rows)
