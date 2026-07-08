"""Core models for QUINTdeepflow3 omit-region quality control."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class OverlayChannelInfo:
    """One plane definition from the overlay channel-map workbook."""

    channel_index: int
    channel_name: str
    content: str
    source_channel: str = ""
    notes: str = ""

    @property
    def plane_index(self) -> int:
        """Zero-based plane index inside the TIFF stack."""

        return max(0, int(self.channel_index) - 1)

    @property
    def is_raw_image(self) -> bool:
        """Whether this plane is a raw-image intensity plane."""

        return self.content.endswith("_raw_image")

    @property
    def is_cell_roi(self) -> bool:
        """Whether this plane is a per-channel cell ROI outline."""

        return self.content.endswith("_cell_roi_outline")

    @property
    def is_overlap_roi(self) -> bool:
        """Whether this plane is a matched multi-channel ROI outline."""

        return self.content == "matched_overlap_roi_outline"

    @property
    def is_outline(self) -> bool:
        """Whether this plane is the atlas outline/edge plane."""

        return self.content == "brain_region_outline_edge_roi"

    @property
    def is_omit_mask(self) -> bool:
        """Whether this plane stores a per-pixel omit-region mask."""

        return self.content == "omit_region_mask"

    @property
    def is_display_code(self) -> bool:
        """Whether this plane stores per-pixel atlas display codes."""

        return self.content == "atlas_display_code"


@dataclass(slots=True)
class OverlaySliceInfo:
    """One overlay TIFF file within the chosen overlay directory."""

    overlay_path: Path
    animal_id: str
    section_id: str
    display_name: str

    @property
    def key(self) -> str:
        """Stable per-slice key used in persisted session files."""

        return self.overlay_path.name


@dataclass(slots=True)
class RegionInfo:
    """Atlas region metadata decoded from a display code."""

    display_code: int
    region_id: int
    region_name: str
    hemisphere: str
    parent_region_id: int | None = None
    hierarchy: str = ""


@dataclass(slots=True)
class OverlayDataset:
    """Loaded overlay stack metadata and atlas codebook."""

    overlay_dir: Path
    channel_map_path: Path
    codebook_path: Path | None
    slices: list[OverlaySliceInfo]
    channels: list[OverlayChannelInfo]
    region_lookup: dict[int, RegionInfo]

    @property
    def visible_channel_infos(self) -> list[OverlayChannelInfo]:
        """Displayable planes excluding the hidden atlas-code plane."""

        return [info for info in self.channels if not info.is_display_code]

    @property
    def raw_contents(self) -> list[str]:
        """Raw-image content identifiers in workbook order."""

        return [info.content for info in self.channels if info.is_raw_image]


@dataclass(slots=True, frozen=True)
class OmitRegionSelection:
    """One connected omitted atlas patch inside a slice."""

    display_code: int
    component_label: int


@dataclass(slots=True)
class OmitSessionState:
    """Persisted interactive selection state for QUINTdeepflow3."""

    overlay_dir: Path
    channel_map_path: Path
    selected_slice_key: str = ""
    visible_contents: list[str] = field(default_factory=list)
    omitted_regions_by_slice: dict[str, list[OmitRegionSelection]] = field(default_factory=dict)
