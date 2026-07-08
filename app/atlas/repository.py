"""Atlas volume loading, hierarchy parsing, and coordinate conversions."""

from __future__ import annotations

import gzip
import json
import logging
import math
import shutil
import struct
from pathlib import Path
import zlib

import numpy as np

from config.settings import AtlasConfig
from data_models.models import AtlasRegion, RegistrationSlice

LOG = logging.getLogger(__name__)


class AtlasRepository:
    """Loads Allen-like atlas volumes and serves region / coordinate lookups."""

    def __init__(self, config: AtlasConfig) -> None:
        self.config = config
        self.labels: np.ndarray | None = None
        self.regions: dict[int, AtlasRegion] = {}
        self._loaded = False

    @property
    def voxel_size_um(self) -> float:
        return self.config.voxel_size_um

    @property
    def midline_quicknii_x(self) -> float:
        """QuickNII X voxel coordinate that corresponds to ML = 0."""

        return self.config.allen_bregma_um[2] / self.config.voxel_size_um

    def load(self) -> None:
        """Load both the label volume and atlas hierarchy."""

        if self._loaded:
            return
        self.labels = self._load_labels(self.config.labels_path)
        configured_resolution = tuple(int(value) for value in self.config.quicknii_resolution_vox)
        label_resolution = tuple(int(value) for value in self.labels.shape)
        if configured_resolution != label_resolution:
            LOG.warning(
                "Atlas resolution mismatch: config quicknii_resolution_vox=%s but labels shape=%s",
                configured_resolution,
                label_resolution,
            )
        self.regions = self._load_tree(self.config.tree_path)
        self._loaded = True
        LOG.info("Loaded atlas %s with %d regions", self.config.name, len(self.regions))

    def require_labels(self) -> np.ndarray:
        """Return the label volume, ensuring it is loaded."""

        self.load()
        assert self.labels is not None
        return self.labels

    def region_for_id(self, region_id: int) -> AtlasRegion | None:
        """Look up region metadata by atlas id."""

        self.load()
        return self.regions.get(int(region_id))

    def region_name(self, region_id: int) -> str:
        """Best-effort region name lookup."""

        region = self.region_for_id(region_id)
        return region.name if region else "Unassigned"

    def region_color(self, region_id: int) -> tuple[int, int, int]:
        """Best-effort region color lookup."""

        region = self.region_for_id(region_id)
        return region.color_rgb if region else (0, 255, 255)

    def sample_labels(self, points_xyz: np.ndarray) -> np.ndarray:
        """Nearest-neighbor sample atlas labels at QuickNII voxel coordinates."""

        labels = self.require_labels()
        coords = np.rint(points_xyz).astype(np.int32, copy=False)
        sampled = np.zeros(coords.shape[:-1], dtype=np.uint32)
        valid = (
            (coords[..., 0] >= 0)
            & (coords[..., 0] < labels.shape[0])
            & (coords[..., 1] >= 0)
            & (coords[..., 1] < labels.shape[1])
            & (coords[..., 2] >= 0)
            & (coords[..., 2] < labels.shape[2])
        )
        if np.any(valid):
            valid_coords = coords[valid]
            sampled[valid] = labels[
                valid_coords[:, 0],
                valid_coords[:, 1],
                valid_coords[:, 2],
            ]
        return sampled

    def sample_labels_local_signed_distance(
        self,
        points_xyz: np.ndarray,
        radius_vox: int = 2,
        batch_size: int = 8192,
    ) -> np.ndarray:
        """Approximate 3D signed-distance sampling from a local voxel neighborhood."""

        if radius_vox < 1:
            return self.sample_labels(points_xyz)

        labels = self.require_labels()
        points = np.asarray(points_xyz, dtype=np.float64)
        original_shape = points.shape[:-1]
        flat_points = points.reshape(-1, 3)
        if flat_points.size == 0:
            return np.zeros(original_shape, dtype=np.uint32)

        offsets = np.stack(
            np.meshgrid(
                np.arange(-radius_vox, radius_vox + 1, dtype=np.int32),
                np.arange(-radius_vox, radius_vox + 1, dtype=np.int32),
                np.arange(-radius_vox, radius_vox + 1, dtype=np.int32),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        corner_offsets = np.stack(
            np.meshgrid(
                np.array([0, 1], dtype=np.int32),
                np.array([0, 1], dtype=np.int32),
                np.array([0, 1], dtype=np.int32),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)

        sampled_flat = np.zeros(flat_points.shape[0], dtype=np.uint32)
        max_local_distance = float(np.sqrt(3.0) * (radius_vox + 1))

        for batch_start in range(0, flat_points.shape[0], max(1, batch_size)):
            batch_end = min(batch_start + max(1, batch_size), flat_points.shape[0])
            chunk_points = flat_points[batch_start:batch_end]
            base = np.floor(chunk_points).astype(np.int32, copy=False)

            neighbor_coords = base[:, None, :] + offsets[None, :, :]
            valid_neighbors = (
                (neighbor_coords[..., 0] >= 0)
                & (neighbor_coords[..., 0] < labels.shape[0])
                & (neighbor_coords[..., 1] >= 0)
                & (neighbor_coords[..., 1] < labels.shape[1])
                & (neighbor_coords[..., 2] >= 0)
                & (neighbor_coords[..., 2] < labels.shape[2])
            )
            neighbor_labels = np.zeros(valid_neighbors.shape, dtype=np.uint32)
            if np.any(valid_neighbors):
                valid_coords = neighbor_coords[valid_neighbors]
                neighbor_labels[valid_neighbors] = labels[
                    valid_coords[:, 0],
                    valid_coords[:, 1],
                    valid_coords[:, 2],
                ]

            corner_coords = base[:, None, :] + corner_offsets[None, :, :]
            valid_corners = (
                (corner_coords[..., 0] >= 0)
                & (corner_coords[..., 0] < labels.shape[0])
                & (corner_coords[..., 1] >= 0)
                & (corner_coords[..., 1] < labels.shape[1])
                & (corner_coords[..., 2] >= 0)
                & (corner_coords[..., 2] < labels.shape[2])
            )
            corner_labels = np.zeros(valid_corners.shape, dtype=np.uint32)
            if np.any(valid_corners):
                valid_corner_coords = corner_coords[valid_corners]
                corner_labels[valid_corners] = labels[
                    valid_corner_coords[:, 0],
                    valid_corner_coords[:, 1],
                    valid_corner_coords[:, 2],
                ]

            distances = np.sqrt(np.sum((neighbor_coords.astype(np.float32) - chunk_points[:, None, :]) ** 2, axis=2))
            candidate_valid = corner_labels > 0
            same_label = neighbor_labels[:, None, :] == corner_labels[:, :, None]
            neighbor_valid = valid_neighbors[:, None, :]
            positive_neighbor = neighbor_labels[:, None, :] > 0

            d_in = np.where(same_label & neighbor_valid, distances[:, None, :], np.inf).min(axis=2)
            d_out = np.where((~same_label) & neighbor_valid & positive_neighbor, distances[:, None, :], np.inf).min(axis=2)
            d_out = np.where(np.isfinite(d_out), d_out, max_local_distance)
            scores = d_out - d_in
            scores[~candidate_valid] = -np.inf

            best_candidate_index = np.argmax(scores, axis=1)
            best_labels = corner_labels[np.arange(chunk_points.shape[0]), best_candidate_index]
            unresolved = best_labels == 0
            if np.any(unresolved):
                best_labels[unresolved] = self.sample_labels(chunk_points[unresolved]).astype(np.uint32, copy=False)
            sampled_flat[batch_start:batch_end] = best_labels

        return sampled_flat.reshape(original_shape)

    def sample_labels_with_mode(
        self,
        points_xyz: np.ndarray,
        mode: str = "nearest",
        radius_vox: int = 2,
        batch_size: int = 8192,
    ) -> np.ndarray:
        """Sample atlas labels using the requested spatial smoothing strategy."""

        normalized = (mode or "nearest").strip().lower()
        if normalized in {"nearest", "nn"}:
            return self.sample_labels(points_xyz)
        if normalized in {"local_signed_distance", "signed_distance", "local_sdf"}:
            return self.sample_labels_local_signed_distance(points_xyz, radius_vox=radius_vox, batch_size=batch_size)
        raise ValueError(f"Unsupported atlas sampling mode: {mode}")

    def adapted_registration_slice(
        self,
        registration_slice: RegistrationSlice,
    ) -> tuple[RegistrationSlice, tuple[float, float, float], bool]:
        """Scale registration anchoring when the atlas voxel grid differs from the JSON target grid."""

        self.load()
        target_resolution = tuple(int(value) for value in (self.labels.shape if self.labels is not None else self.config.quicknii_resolution_vox))
        return registration_slice.scaled_for_target_resolution(target_resolution)

    def quicknii_to_allen_um(self, point_xyz: np.ndarray) -> np.ndarray:
        """Convert QuickNII RAS voxels to Allen PIR micrometers."""

        resolution = 1.0 / self.voxel_size_um
        x = (self.config.quicknii_resolution_vox[1] - 1 - point_xyz[..., 1]) / resolution
        y = (self.config.quicknii_resolution_vox[2] - 1 - point_xyz[..., 2]) / resolution
        z = point_xyz[..., 0] / resolution
        return np.stack((x, y, z), axis=-1)

    def allen_um_to_bregma(
        self,
        point_um: np.ndarray,
        unit: str = "mm",
    ) -> tuple[float, float, float]:
        """Convert Allen PIR micrometers to AP / ML / DV relative to Bregma."""

        bregma_x, bregma_y, bregma_z = self.config.allen_bregma_um
        ap_um = bregma_x - float(point_um[0])
        dv_um = bregma_y - float(point_um[1])
        ml_um = float(point_um[2]) - bregma_z

        if self.config.stereotactic_tilt_deg:
            theta = math.radians(self.config.stereotactic_tilt_deg)
            rotated_ap = ap_um * math.cos(theta) - dv_um * math.sin(theta)
            rotated_dv = ap_um * math.sin(theta) + dv_um * math.cos(theta)
            ap_um = rotated_ap
            dv_um = rotated_dv
        dv_um *= self.config.dv_scale

        scale = 1000.0 if unit.lower() == "mm" else 1.0
        return ap_um / scale, ml_um / scale, dv_um / scale

    def allen_um_to_bregma_array(
        self,
        point_um: np.ndarray,
        unit: str = "mm",
    ) -> np.ndarray:
        """Vectorized conversion from Allen PIR micrometers to AP / ML / DV relative to Bregma."""

        points = np.asarray(point_um, dtype=np.float64)
        bregma_x, bregma_y, bregma_z = self.config.allen_bregma_um
        ap_um = bregma_x - points[..., 0]
        dv_um = bregma_y - points[..., 1]
        ml_um = points[..., 2] - bregma_z

        if self.config.stereotactic_tilt_deg:
            theta = math.radians(self.config.stereotactic_tilt_deg)
            rotated_ap = ap_um * math.cos(theta) - dv_um * math.sin(theta)
            rotated_dv = ap_um * math.sin(theta) + dv_um * math.cos(theta)
            ap_um = rotated_ap
            dv_um = rotated_dv
        dv_um = dv_um * self.config.dv_scale

        scale = 1000.0 if unit.lower() == "mm" else 1.0
        return np.stack((ap_um / scale, ml_um / scale, dv_um / scale), axis=-1)

    def quicknii_ml_um(self, quicknii_x: float) -> float:
        """Convert QuickNII X into Bregma-centered ML in micrometers."""

        return (quicknii_x * self.voxel_size_um) - self.config.allen_bregma_um[2]

    def quicknii_ml_um_array(self, quicknii_x: np.ndarray) -> np.ndarray:
        """Vectorized QuickNII X to Bregma-centered ML in micrometers."""

        return (np.asarray(quicknii_x, dtype=np.float64) * self.voxel_size_um) - self.config.allen_bregma_um[2]

    def _load_tree(self, tree_path: Path) -> dict[int, AtlasRegion]:
        with tree_path.open("r", encoding="utf-8") as handle:
            roots = json.load(handle)

        regions: dict[int, AtlasRegion] = {}

        def visit(node: dict, parent_id: int | None, hierarchy_ids: list[int], hierarchy_names: list[str]) -> None:
            region_id = int(node["id"])
            color_hex = str(node.get("color", "00FFFF"))
            if len(color_hex) != 6:
                color_hex = "00FFFF"
            color_rgb = tuple(int(color_hex[index : index + 2], 16) for index in (0, 2, 4))
            region = AtlasRegion(
                region_id=region_id,
                name=node["name"],
                color_rgb=color_rgb,
                parent_id=parent_id,
                children=[int(child["id"]) for child in node.get("children", [])],
                hierarchy_ids=[*hierarchy_ids, region_id],
                hierarchy_names=[*hierarchy_names, node["name"]],
            )
            regions[region_id] = region
            for child in node.get("children", []):
                visit(child, region_id, region.hierarchy_ids, region.hierarchy_names)

        for root in roots:
            visit(root, None, [], [])
        return regions

    def _load_labels(self, path: Path) -> np.ndarray:
        suffixes = {suffix.lower() for suffix in path.suffixes}
        if ".nrrd" in suffixes or ".nhdr" in suffixes:
            return self._load_nrrd_labels(path)
        return self._load_nifti_labels(path)

    def _load_nifti_labels(self, path: Path) -> np.ndarray:
        """Read a small subset of NIfTI-1 directly so nibabel is optional."""

        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as handle:
            header = handle.read(352)
            sizeof_hdr = struct.unpack("<I", header[:4])[0]
            if sizeof_hdr != 348:
                raise ValueError(f"Unsupported NIfTI header size in {path}: {sizeof_hdr}")
            dims = struct.unpack("<8h", header[40:56])
            datatype = struct.unpack("<h", header[70:72])[0]
            bitpix = struct.unpack("<h", header[72:74])[0]
            vox_offset = int(struct.unpack("<f", header[108:112])[0])
            shape = tuple(int(value) for value in dims[1 : dims[0] + 1])
            dtype_map = {
                2: np.uint8,
                4: np.int16,
                8: np.int32,
                16: np.float32,
                64: np.float64,
                256: np.int8,
                512: np.uint16,
                768: np.uint32,
            }
            if datatype not in dtype_map:
                raise ValueError(f"Unsupported NIfTI datatype {datatype} in {path}")
            handle.seek(vox_offset)
            count = int(np.prod(shape))
            data = np.frombuffer(handle.read(count * (bitpix // 8)), dtype=dtype_map[datatype], count=count)
        # NIfTI stores voxel data with the first axis varying fastest on disk, so
        # direct C-order reshape scrambles the Allen label volume. Fortran-order
        # reconstruction matches QuickNII / Nutil region IDs.
        return data.reshape(shape, order="F")

    def _load_nrrd_labels(self, path: Path) -> np.ndarray:
        """Read a minimal subset of NRRD files used by Allen annotation volumes."""

        with path.open("rb") as handle:
            header_lines: list[bytes] = []
            while True:
                line = handle.readline()
                if not line:
                    raise ValueError(f"NRRD header ended unexpectedly in {path}")
                header_lines.append(line)
                if line in {b"\n", b"\r\n"}:
                    break
            data_offset = handle.tell()

        header: dict[str, str] = {}
        for raw_line in header_lines:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line or line.startswith("#") or line.startswith("NRRD"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            header[key.strip().lower()] = value.strip()

        dimension = int(header.get("dimension", "0"))
        if dimension < 1:
            raise ValueError(f"Unsupported NRRD dimension in {path}: {dimension}")
        sizes = tuple(int(value) for value in header.get("sizes", "").split())
        if len(sizes) != dimension:
            raise ValueError(f"Invalid NRRD sizes in {path}: {sizes}")

        dtype_map = {
            "uchar": np.uint8,
            "uint8": np.uint8,
            "signed char": np.int8,
            "int8": np.int8,
            "short": np.int16,
            "short int": np.int16,
            "int16": np.int16,
            "ushort": np.uint16,
            "unsigned short": np.uint16,
            "uint16": np.uint16,
            "int": np.int32,
            "int32": np.int32,
            "uint": np.uint32,
            "unsigned int": np.uint32,
            "uint32": np.uint32,
            "float": np.float32,
            "double": np.float64,
        }
        type_name = header.get("type", "").lower()
        if type_name not in dtype_map:
            raise ValueError(f"Unsupported NRRD type in {path}: {type_name}")
        dtype = np.dtype(dtype_map[type_name])

        encoding = header.get("encoding", "raw").lower()
        if encoding not in {"gzip", "gz", "raw", "zlib"}:
            raise ValueError(f"Unsupported NRRD encoding in {path}: {encoding}")

        endian = header.get("endian", "little").lower()
        if dtype.itemsize > 1:
            if endian == "little":
                dtype = dtype.newbyteorder("<")
            elif endian == "big":
                dtype = dtype.newbyteorder(">")
            else:
                raise ValueError(f"Unsupported NRRD endian in {path}: {endian}")

        count = int(np.prod(sizes))
        expected_bytes = count * dtype.itemsize
        array = self._load_nrrd_array(
            path=path,
            dtype=dtype,
            sizes=sizes,
            data_offset=data_offset,
            expected_bytes=expected_bytes,
            encoding=encoding,
        )
        kinds = [value.strip().lower() for value in header.get("kinds", "").split()]
        if kinds and kinds[0] in {"list", "vector", "rgb-color", "rgba-color"}:
            array = np.moveaxis(array, 0, -1)
        if array.ndim == 3:
            array = self._orient_allen_annotation_volume(array)
        return array.astype(np.uint32, copy=False)

    def _load_nrrd_array(
        self,
        path: Path,
        dtype: np.dtype,
        sizes: tuple[int, ...],
        data_offset: int,
        expected_bytes: int,
        encoding: str,
    ) -> np.ndarray:
        """Load or materialize NRRD payload into an array, using memmap for large volumes."""

        normalized_encoding = encoding.lower()
        if normalized_encoding == "raw":
            return np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=sizes, order="F")

        cache_path = path.with_name(f"{path.stem}__quintnext_cache.raw")
        if not cache_path.exists() or cache_path.stat().st_size != expected_bytes:
            LOG.info("Materializing compressed NRRD payload to %s", cache_path)
            self._materialize_nrrd_payload(path, cache_path, data_offset, normalized_encoding)
        return np.memmap(cache_path, dtype=dtype, mode="r", shape=sizes, order="F")

    def _materialize_nrrd_payload(
        self,
        source_path: Path,
        cache_path: Path,
        data_offset: int,
        encoding: str,
    ) -> None:
        """Expand a compressed NRRD payload into a raw cache file for disk-backed access."""

        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with source_path.open("rb") as source_handle:
            source_handle.seek(data_offset)
            with tmp_path.open("wb") as out_handle:
                if encoding in {"gzip", "gz"}:
                    with gzip.GzipFile(fileobj=source_handle) as gzip_handle:
                        shutil.copyfileobj(gzip_handle, out_handle, length=8 * 1024 * 1024)
                elif encoding == "zlib":
                    decompressor = zlib.decompressobj()
                    while True:
                        chunk = source_handle.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        out_handle.write(decompressor.decompress(chunk))
                    out_handle.write(decompressor.flush())
                else:
                    raise ValueError(f"Unsupported NRRD encoding in {source_path}: {encoding}")
        tmp_path.replace(cache_path)

    def _orient_allen_annotation_volume(self, array: np.ndarray) -> np.ndarray:
        """Match official Allen NRRD axis order to the QuickNII/Nutil orientation used locally."""

        if array.ndim != 3:
            return array
        oriented = np.transpose(array, (2, 0, 1))
        oriented = np.flip(oriented, axis=(1, 2))
        return oriented
