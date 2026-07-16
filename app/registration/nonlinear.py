"""VisuAlign-style nonlinear in-plane correction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from data_models.models import RegistrationSlice

PADDING_MARKER_RATIO = 0.10
MIN_TRIANGLE_AREA2 = 1.0e-6


def _marker_xy(x: float, y: float) -> list[float]:
    return [float(x), float(y), float(x), float(y)]


@dataclass(slots=True)
class VisualignTriangle:
    """Exact piecewise-affine triangle used by VisuAlign."""

    a: int
    b: int
    c: int
    trimarkers: list[list[float]]
    source_points: np.ndarray
    target_points: np.ndarray
    minx: float
    miny: float
    maxx: float
    maxy: float
    decomp: np.ndarray | None
    den: float
    mdenx: float
    mdeny: float
    r2den: float
    source_a: np.ndarray
    source_b_delta: np.ndarray
    source_c_delta: np.ndarray
    target_a: np.ndarray
    source_area2: float
    target_area2: float
    orientation_preserving: bool

    def __init__(self, first: int, second: int, third: int, trimarkers: list[list[float]]) -> None:
        indices = sorted((int(first), int(second), int(third)))
        self.a, self.b, self.c = indices
        self.trimarkers = trimarkers
        self.source_points = np.asarray(
            [
                [float(trimarkers[self.a][0]), float(trimarkers[self.a][1])],
                [float(trimarkers[self.b][0]), float(trimarkers[self.b][1])],
                [float(trimarkers[self.c][0]), float(trimarkers[self.c][1])],
            ],
            dtype=np.float64,
        )
        self.target_points = np.asarray(
            [
                [float(trimarkers[self.a][2]), float(trimarkers[self.a][3])],
                [float(trimarkers[self.b][2]), float(trimarkers[self.b][3])],
                [float(trimarkers[self.c][2]), float(trimarkers[self.c][3])],
            ],
            dtype=np.float64,
        )
        self.source_a = self.source_points[0].copy()
        self.source_b_delta = self.source_points[1] - self.source_points[0]
        self.source_c_delta = self.source_points[2] - self.source_points[0]
        self.target_a = self.target_points[0].copy()
        self.source_area2 = self._signed_area2(self.source_points)
        self.target_area2 = self._signed_area2(self.target_points)
        self.orientation_preserving = (
            abs(self.source_area2) > MIN_TRIANGLE_AREA2
            and abs(self.target_area2) > MIN_TRIANGLE_AREA2
            and (self.source_area2 * self.target_area2) > 0.0
        )
        self.minx = float(np.min(self.target_points[:, 0]))
        self.miny = float(np.min(self.target_points[:, 1]))
        self.maxx = float(np.max(self.target_points[:, 0]))
        self.maxy = float(np.max(self.target_points[:, 1]))

        ax, ay = self.target_points[0]
        bx, by = self.target_points[1]
        cx, cy = self.target_points[2]
        matrix = np.asarray(
            [
                [bx - ax, cx - ax],
                [by - ay, cy - ay],
            ],
            dtype=np.float64,
        )
        try:
            self.decomp = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            self.decomp = None

        dab = self._d2(bx, by, cx, cy)
        dac = self._d2(ax, ay, cx, cy)
        dbc = self._d2(ax, ay, bx, by)
        wa = dab * (dac + dbc - dab)
        wb = dac * (dbc + dab - dac)
        wc = dbc * (dab + dac - dbc)
        self.den = float(wa + wb + wc)
        self.mdenx = float((wa * ax) + (wb * bx) + (wc * cx))
        self.mdeny = float((wa * ay) + (wb * by) + (wc * cy))
        self.r2den = self._d2(ax * self.den, ay * self.den, self.mdenx, self.mdeny)

    @staticmethod
    def _d2(x1: float, y1: float, x2: float, y2: float) -> float:
        dx = float(x1 - x2)
        dy = float(y1 - y2)
        return (dx * dx) + (dy * dy)

    @staticmethod
    def _signed_area2(points: np.ndarray) -> float:
        return float(
            ((points[1, 0] - points[0, 0]) * (points[2, 1] - points[0, 1]))
            - ((points[1, 1] - points[0, 1]) * (points[2, 0] - points[0, 0]))
        )

    def intri(self, x: float, y: float) -> np.ndarray | None:
        if x < self.minx or x > self.maxx or y < self.miny or y > self.maxy or self.decomp is None:
            return None
        ax, ay = self.target_points[0]
        bary = self.decomp @ np.asarray([float(x - ax), float(y - ay)], dtype=np.float64)
        u = float(bary[0])
        v = float(bary[1])
        if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0 or (u + v) > 1.0:
            return None
        return np.asarray([u, v], dtype=np.float64)

    def transform(self, x: float, y: float) -> np.ndarray | None:
        bary = self.intri(x, y)
        if bary is None:
            return None
        u = float(bary[0])
        v = float(bary[1])
        return self.source_a + (self.source_b_delta * u) + (self.source_c_delta * v)

    def incirc(self, x: float, y: float) -> bool:
        if not np.isfinite(self.den) or abs(self.den) <= 1.0e-12:
            return False
        return self._d2(float(x) * self.den, float(y) * self.den, self.mdenx, self.mdeny) < self.r2den


@dataclass(slots=True)
class VisualignPiecewiseAffineMapper:
    """Inverse mapper from adjusted image pixels to the affine registration plane."""

    trimarkers: list[list[float]]
    triangles: list[VisualignTriangle]
    registration_size: tuple[int, int]

    def transform_point(self, target_xy: np.ndarray | tuple[float, float] | list[float]) -> np.ndarray | None:
        point = np.asarray(target_xy, dtype=np.float64).reshape(2)
        for triangle in self.triangles:
            transformed = triangle.transform(float(point[0]), float(point[1]))
            if transformed is not None:
                return transformed
        return None

    def map_target_to_source(self, target_points_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(target_points_xy, dtype=np.float64)
        original_shape = points.shape
        flat_points = points.reshape(-1, 2)
        mapped = flat_points.copy()
        assigned = np.zeros(flat_points.shape[0], dtype=bool)
        for triangle in self.triangles:
            if np.all(assigned):
                break
            remaining = ~assigned
            candidate_idx = np.flatnonzero(remaining)
            candidate_points = flat_points[remaining]
            if candidate_points.size == 0:
                break
            bbox_mask = (
                (candidate_points[:, 0] >= triangle.minx)
                & (candidate_points[:, 0] <= triangle.maxx)
                & (candidate_points[:, 1] >= triangle.miny)
                & (candidate_points[:, 1] <= triangle.maxy)
            )
            if not np.any(bbox_mask):
                continue
            bbox_idx = candidate_idx[bbox_mask]
            bbox_points = candidate_points[bbox_mask]
            if triangle.decomp is None:
                continue
            delta = bbox_points - triangle.target_a
            bary = delta @ triangle.decomp.T
            inside = (
                (bary[:, 0] >= 0.0)
                & (bary[:, 0] <= 1.0)
                & (bary[:, 1] >= 0.0)
                & (bary[:, 1] <= 1.0)
                & ((bary[:, 0] + bary[:, 1]) <= 1.0)
            )
            if not np.any(inside):
                continue
            inside_idx = bbox_idx[inside]
            inside_bary = bary[inside]
            mapped[inside_idx] = (
                triangle.source_a
                + (triangle.source_b_delta * inside_bary[:, [0]])
                + (triangle.source_c_delta * inside_bary[:, [1]])
            )
            assigned[inside_idx] = True
        return mapped.reshape(original_shape)


def _padding_markers(width: float, height: float) -> list[list[float]]:
    return [
        _marker_xy(-PADDING_MARKER_RATIO * width, -PADDING_MARKER_RATIO * height),
        _marker_xy((1.0 + PADDING_MARKER_RATIO) * width, -PADDING_MARKER_RATIO * height),
        _marker_xy(-PADDING_MARKER_RATIO * width, (1.0 + PADDING_MARKER_RATIO) * height),
        _marker_xy((1.0 + PADDING_MARKER_RATIO) * width, (1.0 + PADDING_MARKER_RATIO) * height),
    ]


def _triangulate_markers(markers: list[list[float]], width: float, height: float) -> tuple[list[list[float]], list[VisualignTriangle]]:
    trimarkers = _padding_markers(width, height)
    triangles = [
        VisualignTriangle(0, 1, 2, trimarkers),
        VisualignTriangle(1, 2, 3, trimarkers),
    ]

    for marker in markers:
        if len(marker) < 4:
            continue
        px = float(marker[2])
        py = float(marker[3])
        inside_existing = False
        bad_triangles: list[VisualignTriangle] = []
        for triangle in list(triangles):
            if not inside_existing and triangle.intri(px, py) is not None:
                inside_existing = True
            if triangle.incirc(px, py):
                bad_triangles.append(triangle)
        if not inside_existing:
            continue

        edge_counts: dict[tuple[int, int], int] = {}
        for triangle in bad_triangles:
            for edge in ((triangle.a, triangle.b), (triangle.a, triangle.c), (triangle.b, triangle.c)):
                edge_counts[edge] = edge_counts.get(edge, 0) - 1

        trial_markers = [*trimarkers, [float(value) for value in marker[:4]]]
        new_index = len(trial_markers) - 1
        new_triangles: list[VisualignTriangle] = []
        rejected_folded_triangle = False
        for edge, count in edge_counts.items():
            if count == -1:
                candidate = VisualignTriangle(edge[0], edge[1], new_index, trial_markers)
                if candidate.decomp is not None and candidate.orientation_preserving:
                    new_triangles.append(candidate)
                else:
                    rejected_folded_triangle = True
        if rejected_folded_triangle or not new_triangles:
            continue
        triangles = [triangle for triangle in triangles if triangle not in bad_triangles]
        triangles.extend(new_triangles)
        trimarkers = trial_markers
    return trimarkers, triangles


def build_marker_inverse_warp(
    registration_slice: RegistrationSlice,
    image_shape: tuple[int, int],
    add_corner_anchors: bool = True,
) -> VisualignPiecewiseAffineMapper | None:
    """Build the same piecewise-affine marker warp used by VisuAlign."""

    registration_height = int(registration_slice.height or image_shape[0])
    registration_width = int(registration_slice.width or image_shape[1])
    marker_rows = [
        [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
        for row in registration_slice.markers
        if len(row) >= 4
    ]
    marker_key = tuple(tuple(float(value) for value in row) for row in marker_rows)
    return _build_marker_inverse_warp_cached(registration_width, registration_height, marker_key)


def build_piecewise_affine_mapper(
    width: int,
    height: int,
    marker_rows: list[list[float]] | tuple[tuple[float, float, float, float], ...],
) -> VisualignPiecewiseAffineMapper | None:
    """Build an image-space piecewise-affine mapper from arbitrary marker rows."""

    marker_key = tuple(tuple(float(value) for value in row[:4]) for row in marker_rows if len(row) >= 4)
    return _build_marker_inverse_warp_cached(int(width), int(height), marker_key)


@lru_cache(maxsize=256)
def _build_marker_inverse_warp_cached(
    registration_width: int,
    registration_height: int,
    marker_key: tuple[tuple[float, float, float, float], ...],
) -> VisualignPiecewiseAffineMapper | None:
    marker_rows = [list(row) for row in marker_key]
    trimarkers, triangles = _triangulate_markers(marker_rows, float(registration_width), float(registration_height))
    if not triangles:
        return None
    return VisualignPiecewiseAffineMapper(
        trimarkers=trimarkers,
        triangles=triangles,
        registration_size=(registration_height, registration_width),
    )


def image_points_to_registration_source(
    points_xy: np.ndarray,
    image_shape: tuple[int, int],
    registration_slice: RegistrationSlice,
    mapper: VisualignPiecewiseAffineMapper | None = None,
    image_offset_px: tuple[float, float] = (0.0, 0.0),
    source_image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Convert image pixel coordinates into affine registration pixel coordinates."""

    points = np.asarray(points_xy, dtype=np.float64).copy()
    registration_height = float(registration_slice.height or image_shape[0])
    registration_width = float(registration_slice.width or image_shape[1])
    effective_image_shape = source_image_shape or image_shape
    image_height = float(max(effective_image_shape[0], 1))
    image_width = float(max(effective_image_shape[1], 1))
    offset_x = float(image_offset_px[0])
    offset_y = float(image_offset_px[1])

    points[..., 0] = (points[..., 0] - offset_x) * (registration_width / image_width)
    points[..., 1] = (points[..., 1] - offset_y) * (registration_height / image_height)

    if mapper is None:
        return points
    return mapper.map_target_to_source(points)
