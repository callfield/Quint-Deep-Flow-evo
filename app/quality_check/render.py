"""Overlay rendering helpers for QUINTdeepflow3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage

from quality_check.models import OmitRegionSelection, OverlayChannelInfo


RAW_CHANNEL_RGB = {
    "CH1": np.asarray([90, 220, 255], dtype=np.float32),
    "CH2": np.asarray([255, 110, 110], dtype=np.float32),
    "CH3": np.asarray([130, 255, 150], dtype=np.float32),
    "CH4": np.asarray([255, 195, 90], dtype=np.float32),
}
CELL_DEFAULT_RGB = np.asarray([245, 245, 245], dtype=np.float32)
OVERLAP_RGB = np.asarray([255, 70, 210], dtype=np.float32)
OUTLINE_RGB = np.asarray([150, 210, 255], dtype=np.float32)
OMIT_RGB = np.asarray([255, 220, 60], dtype=np.float32)
OMIT_MASK_RGB = np.asarray([255, 64, 64], dtype=np.float32)


@dataclass(slots=True)
class CanvasTransform:
    """Mapping between original image pixels and fitted canvas pixels."""

    image_width: int
    image_height: int
    draw_width: int
    draw_height: int
    offset_x: int
    offset_y: int
    scale: float

    def contains(self, x_canvas: float, y_canvas: float) -> bool:
        """Whether a canvas-space point lands inside the drawn image."""

        return (
            self.offset_x <= x_canvas < self.offset_x + self.draw_width
            and self.offset_y <= y_canvas < self.offset_y + self.draw_height
        )

    def to_image_xy(self, x_canvas: float, y_canvas: float) -> tuple[int, int] | None:
        """Convert a canvas-space point into original image coordinates."""

        if not self.contains(x_canvas, y_canvas):
            return None
        x_image = int(np.clip((x_canvas - self.offset_x) / max(self.scale, 1e-8), 0, self.image_width - 1))
        y_image = int(np.clip((y_canvas - self.offset_y) / max(self.scale, 1e-8), 0, self.image_height - 1))
        return x_image, y_image


def compute_canvas_transform(
    image_shape: tuple[int, int],
    canvas_width: int,
    canvas_height: int,
    zoom_factor: float | None = None,
) -> CanvasTransform:
    """Return a transform for one image."""

    height, width = image_shape
    safe_canvas_width = max(1, int(canvas_width))
    safe_canvas_height = max(1, int(canvas_height))
    if zoom_factor is None:
        scale = min(safe_canvas_width / max(width, 1), safe_canvas_height / max(height, 1))
        scale = min(scale, 1.0) if scale > 0 else 1.0
    else:
        scale = max(0.05, float(zoom_factor))
    draw_width = max(1, int(round(width * scale)))
    draw_height = max(1, int(round(height * scale)))
    return CanvasTransform(
        image_width=width,
        image_height=height,
        draw_width=draw_width,
        draw_height=draw_height,
        offset_x=0,
        offset_y=0,
        scale=scale,
    )


def render_qc_image(
    stack: np.ndarray,
    channels: list[OverlayChannelInfo],
    visible_contents: list[str],
    omitted_regions: list[OmitRegionSelection],
) -> Image.Image:
    """Compose a human-readable QC image from one overlay TIFF stack."""

    atlas_plane = _atlas_display_plane(stack, channels)
    height, width = atlas_plane.shape
    rgb = _compose_raw_rgb(stack, channels, visible_contents, output_shape=(height, width))

    for info in channels:
        if info.content not in visible_contents:
            continue
        plane = stack[info.plane_index]
        if info.is_cell_roi:
            color = _channel_rgb(info.source_channel)
            rgb = _apply_mask_overlay(rgb, plane > 0, color=color, alpha=0.88)
        elif info.is_overlap_roi:
            rgb = _apply_mask_overlay(rgb, plane > 0, color=OVERLAP_RGB, alpha=0.92)
        elif info.is_outline:
            outline_alpha = _normalize_outline_alpha(plane)
            rgb = _apply_alpha_overlay(rgb, outline_alpha, OUTLINE_RGB, alpha=0.95)
        elif info.is_omit_mask:
            omit_alpha = _normalize_omit_alpha(plane)
            rgb = _apply_alpha_overlay(rgb, omit_alpha, OMIT_MASK_RGB, alpha=0.60)

    if omitted_regions:
        selected_boundary = _selected_region_boundary_mask(atlas_plane, omitted_regions)
        if np.any(selected_boundary):
            rgb = _apply_mask_overlay(rgb, selected_boundary, color=OMIT_RGB, alpha=1.0)

    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def display_code_at_xy(stack: np.ndarray, channels: list[OverlayChannelInfo], x_px: int, y_px: int) -> int:
    """Return the atlas display code at one image-space pixel."""

    atlas_plane = _atlas_display_plane(stack, channels)
    return int(atlas_plane[int(y_px), int(x_px)])


def component_selection_at_xy(
    stack: np.ndarray,
    channels: list[OverlayChannelInfo],
    x_px: int,
    y_px: int,
) -> OmitRegionSelection | None:
    """Return the clicked connected atlas patch instead of the whole display code."""

    atlas_plane = _atlas_display_plane(stack, channels)
    display_code = int(atlas_plane[int(y_px), int(x_px)])
    if display_code == 0:
        return None
    component_labels = component_label_map_for_code(atlas_plane, display_code)
    component_label = int(component_labels[int(y_px), int(x_px)])
    if component_label <= 0:
        return None
    return OmitRegionSelection(display_code=display_code, component_label=component_label)


def component_label_map_for_code(atlas_plane: np.ndarray, display_code: int) -> np.ndarray:
    """Return labeled connected components for one atlas display code."""

    mask = np.asarray(atlas_plane == int(display_code), dtype=np.uint8)
    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.int32)
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return labels.astype(np.int32, copy=False)


def _compose_raw_rgb(
    stack: np.ndarray,
    channels: list[OverlayChannelInfo],
    visible_contents: list[str],
    output_shape: tuple[int, int],
) -> np.ndarray:
    raw_infos = [info for info in channels if info.is_raw_image and info.content in visible_contents]
    height, width = output_shape
    if not raw_infos:
        return np.zeros((height, width, 3), dtype=np.float32)

    if len(raw_infos) == 1:
        plane = _scale_to_uint8(stack[raw_infos[0].plane_index])
        return np.repeat(plane[..., None], 3, axis=2).astype(np.float32)

    composed = np.zeros((height, width, 3), dtype=np.float32)
    for info in raw_infos:
        plane = _scale_to_unit(stack[info.plane_index])
        color = _channel_rgb(info.source_channel) / 255.0
        composed += plane[..., None] * color[None, None, :] * 255.0
    return np.clip(composed, 0, 255)


def _scale_to_uint8(plane: np.ndarray) -> np.ndarray:
    return np.rint(_scale_to_unit(plane) * 255.0).astype(np.uint8)


def _scale_to_unit(plane: np.ndarray) -> np.ndarray:
    array = np.asarray(plane, dtype=np.float32)
    positive = array[array > 0]
    if positive.size == 0:
        low = float(array.min())
        high = float(array.max())
    else:
        low = float(np.percentile(positive, 1.0))
        high = float(np.percentile(positive, 99.5))
    if not np.isfinite(low):
        low = float(array.min())
    if not np.isfinite(high):
        high = float(array.max())
    if high <= low:
        return np.zeros(array.shape, dtype=np.float32)
    scaled = (array - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0)


def _normalize_outline_alpha(plane: np.ndarray) -> np.ndarray:
    array = np.asarray(plane, dtype=np.float32)
    high = float(array.max())
    if high <= 0:
        return np.zeros(array.shape, dtype=np.float32)
    scaled = np.clip(array / high, 0.0, 1.0)
    return ndimage.maximum_filter(scaled, size=3)


def _normalize_omit_alpha(plane: np.ndarray) -> np.ndarray:
    array = np.asarray(plane, dtype=np.float32)
    high = float(array.max())
    if high <= 0:
        return np.zeros(array.shape, dtype=np.float32)
    return np.clip(array / high, 0.0, 1.0)


def _apply_mask_overlay(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    if not np.any(mask):
        return rgb
    out = rgb.copy()
    out[mask] = out[mask] * (1.0 - float(alpha)) + color[None, :] * float(alpha)
    return out


def _apply_alpha_overlay(rgb: np.ndarray, alpha_mask: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    if not np.any(alpha_mask):
        return rgb
    out = rgb.copy()
    effective = np.clip(alpha_mask.astype(np.float32) * float(alpha), 0.0, 1.0)
    out = out * (1.0 - effective[..., None]) + color[None, None, :] * effective[..., None]
    return out


def _selected_region_boundary_mask(atlas_plane: np.ndarray, selected_regions: list[OmitRegionSelection]) -> np.ndarray:
    selected_mask = np.zeros(atlas_plane.shape, dtype=bool)
    for selection in selected_regions:
        component_labels = component_label_map_for_code(atlas_plane, selection.display_code)
        if selection.component_label <= 0:
            continue
        selected_mask |= component_labels == int(selection.component_label)
    if not np.any(selected_mask):
        return np.zeros(selected_mask.shape, dtype=bool)
    boundary = selected_mask & ~ndimage.binary_erosion(selected_mask)
    return ndimage.binary_dilation(boundary, iterations=3)


def _atlas_display_plane(stack: np.ndarray, channels: list[OverlayChannelInfo]) -> np.ndarray:
    atlas_infos = [info for info in channels if info.is_display_code]
    if not atlas_infos:
        raise KeyError("atlas_display_code plane was not found in the channel map.")
    return np.asarray(stack[atlas_infos[0].plane_index], dtype=np.int64)


def _channel_rgb(channel_name: str) -> np.ndarray:
    return RAW_CHANNEL_RGB.get(str(channel_name).upper(), CELL_DEFAULT_RGB)
