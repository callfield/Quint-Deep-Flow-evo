"""Tkinter atlas fitting GUI for manual atlas alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
import os
from pathlib import Path
import re
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import json

import numpy as np
import PIL.ImageDraw
from PIL import Image, ImageFilter, ImageTk
from scipy.spatial.transform import Rotation
try:
    from skimage.measure import approximate_polygon, find_contours
except Exception:  # pragma: no cover - optional vector contour preview
    approximate_polygon = None
    find_contours = None
import yaml

from app_version import APP_VERSION, version_label
from atlas.repository import AtlasRepository
from config.settings import load_app_config
from data_models.models import RegistrationSlice
from io_utils.image_io import ensure_rgb, load_image_array
from overlays.render import build_registered_maps
from registration.nonlinear import (
    build_marker_inverse_warp,
    build_piecewise_affine_mapper,
    image_points_to_registration_source,
)
from registration.parser import parse_registration_file


SUPPORTED_IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(slots=True)
class SliceFitState:
    center_ml: float
    center_ap: float
    center_dv: float
    span_ml: float
    span_dv: float
    tilt_ml_deg: float = 0.0
    tilt_dv_deg: float = 0.0
    roll_deg: float = 0.0
    registration_width: int | None = None
    registration_height: int | None = None
    origin_vec: tuple[float, float, float] | None = None
    u_vec: tuple[float, float, float] | None = None
    v_vec: tuple[float, float, float] | None = None
    markers: list[list[float]] = field(default_factory=list)
    omit_strokes: list[dict[str, object]] = field(default_factory=list)


class AtlasFittingApp(tk.Tk):
    """Python atlas fitting GUI with QDF1-compatible JSON export."""

    WINDOW_GEOMETRY = "1820x980"
    WINDOW_MINSIZE = (1500, 860)
    DISPLAY_PREVIEW_MAX_EDGE = 960
    DISPLAY_PREVIEW_MAX_EDGE_INTERACTIVE = 360
    DISPLAY_PREVIEW_MAX_EDGE_MARKER = 256
    SIDE_PANEL_WIDTH = 720
    ATLAS_MARGIN_RATIO = 0.30
    OMIT_MASK_DIR_SUFFIX = "_omitMasks"
    OMIT_STATE_SUFFIX = "_omit_state.json"
    BROWSE_HISTORY_FILE = ".QDFevo_2_AtlasFitter_browse_history.json"
    INTERACTIVE_RENDER_DELAY_MS = 8
    FINAL_RENDER_DELAY_MS = 45
    INTERACTIVE_REFINE_DELAY_MS = 180

    def __init__(self, app_name: str = "QDFevo_2_AtlasFitter") -> None:
        super().__init__()
        self.withdraw()
        self.app_name = app_name
        self.title(version_label(app_name))
        self.geometry(self.WINDOW_GEOMETRY)
        self.minsize(*self.WINDOW_MINSIZE)

        self.default_config_path = Path(__file__).resolve().parents[1] / "sample_configs" / "default_config.yaml"
        self.app_config = load_app_config(self.default_config_path)
        self._base_atlas_config = replace(self.app_config.atlas)
        self.atlas = AtlasRepository(self.app_config.atlas)
        self.atlas.load()
        self._browse_history = self._load_browse_history()

        self.image_folder_var = tk.StringVar()
        self.input_json_var = tk.StringVar()
        self.output_json_var = tk.StringVar()
        self.session_path_var = tk.StringVar()
        self.slice_label_var = tk.StringVar(value="No slice loaded")
        self.status_var = tk.StringVar(value="Ready")
        self.opacity_var = tk.DoubleVar(value=0.70)
        self.space_opacity_factor = 0.30
        self.preview_note_var = tk.StringVar(value="")
        self.hover_region_var = tk.StringVar(value="")
        self.display_channel_var = tk.StringVar(value="Auto(JSON)")
        self.marker_mode_var = tk.StringVar(value="pan")
        self.omit_draw_mode_var = tk.StringVar(value="brush")
        self.omit_brush_px_var = tk.DoubleVar(value=150.0)
        self.preview_mode_var = tk.StringVar(value="")

        self.ap_var = tk.StringVar()
        self.center_ml_var = tk.StringVar()
        self.center_dv_var = tk.StringVar()
        self.span_ml_var = tk.StringVar()
        self.span_dv_var = tk.StringVar()
        self.tilt_ml_var = tk.StringVar(value="0.0")
        self.tilt_dv_var = tk.StringVar(value="0.0")
        self.roll_var = tk.StringVar(value="0.0")

        self.image_paths: list[Path] = []
        self.image_sizes: dict[str, tuple[int, int]] = {}
        self.slice_states: dict[str, SliceFitState] = {}
        self.current_index = -1
        self.current_preview_rgb: np.ndarray | None = None
        self.current_preview_shape: tuple[int, int] | None = None
        self.current_preview_scale = 1.0
        self.current_photo: ImageTk.PhotoImage | None = None
        self.current_base_photo: ImageTk.PhotoImage | None = None
        self.current_atlas_photo: ImageTk.PhotoImage | None = None
        self.current_omit_photo: ImageTk.PhotoImage | None = None
        self.current_display_image_path: Path | None = None
        self._current_atlas_alpha: np.ndarray | None = None
        self._current_boundary_points: np.ndarray | None = None
        self._current_boundary_alpha_values: np.ndarray | None = None
        self._current_boundary_contours: list[np.ndarray] = []
        self._current_base_signature: tuple[str, tuple[int, int], tuple[int, int]] | None = None
        self.current_region_map: np.ndarray | None = None
        self.current_display_offset = (0.0, 0.0)
        self.current_image_offset_in_composite = (0.0, 0.0)
        self.current_composite_shape: tuple[int, int] | None = None
        self._canvas_base_item: int | None = None
        self._canvas_atlas_item: int | None = None
        self._canvas_omit_item: int | None = None
        self._marker_canvas_items: list[int] = []
        self._preview_contour_canvas_items: list[int] = []
        self._preview_overlay_shift = (0.0, 0.0)
        self.view_zoom_factor = 1.0
        self._reset_view_on_next_render = False
        self._last_canvas_xy: tuple[float, float] | None = None
        self.current_target_name = self.app_config.atlas.name
        self.current_target_resolution = tuple(int(value) for value in self.app_config.atlas.quicknii_resolution_vox)
        self.saved_slice_states: dict[str, SliceFitState] = {}
        self._render_after_id: str | None = None
        self._refine_after_id: str | None = None
        self._last_loaded_input_json: str | None = None
        self._render_interactive = False
        self._render_in_progress = False
        self._rerender_requested = False
        self._rerender_interactive = False
        self._render_request_id = 0
        self._marker_input_ready = False
        self._preview_mode_active = False
        self._space_opacity_active = False
        self._space_release_after_id: str | None = None
        self._ignore_var_updates = False
        self._drag_mode: str | None = None
        self._drag_start_xy: tuple[int, int] | None = None
        self._drag_start_state: SliceFitState | None = None
        self._drag_overlay_base_alpha: np.ndarray | None = None
        self._drag_overlay_base_state: SliceFitState | None = None
        self._pending_marker_source: np.ndarray | None = None
        self._selected_marker_index: int | None = None
        self._selected_omit_stroke_index: int | None = None
        self._omit_preview_item: int | None = None
        self._drag_tension_mapper = None
        self._live_omit_points: list[list[float]] = []
        self._live_omit_size_px: float = 0.0
        self._live_omit_mode: str = "brush"
        self._live_omit_canvas_item: int | None = None

        self._configure_style()
        self._build_layout()
        self._bind_shortcuts()
        self._stabilize_window()
        self.deiconify()

    def _switch_display_atlas_for_target(self, target_name: str, target_resolution: tuple[int, int, int]) -> None:
        current_resolution = tuple(int(value) for value in self.atlas.config.quicknii_resolution_vox)
        if current_resolution == tuple(int(value) for value in target_resolution):
            return

        atlas_config = replace(self._base_atlas_config)
        lowered_target = str(target_name).lower()
        target_resolution = tuple(int(value) for value in target_resolution)
        labels_dir = Path(atlas_config.labels_path).resolve().parent

        if target_resolution == (456, 528, 320) or "25um" in lowered_target:
            atlas_config.name = "allen_mouse_ccfv3_2017_25um"
            atlas_config.labels_path = labels_dir / "annotation_25.nrrd"
            atlas_config.voxel_size_um = 25.0
            atlas_config.quicknii_resolution_vox = (456, 528, 320)
        else:
            atlas_config = replace(self._base_atlas_config)

        self.atlas = AtlasRepository(atlas_config)
        self.atlas.load()

    def _configure_style(self) -> None:
        self.option_add("*Font", "{Segoe UI} 10")
        self.option_add("*TCombobox*Listbox.font", "{Segoe UI} 10")
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("TButton", padding=(10, 4))
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Hover.TLabel", font=("Segoe UI", 24, "bold"), foreground="#082f9a")

    def _stabilize_window(self) -> None:
        self.update_idletasks()
        self.geometry(self.WINDOW_GEOMETRY)
        self.minsize(*self.WINDOW_MINSIZE)

    def _capture_window_layout_guard(self) -> tuple[str, str, int, int]:
        try:
            window_state = str(self.state())
        except tk.TclError:
            window_state = "normal"
        try:
            geometry = str(self.geometry())
        except tk.TclError:
            geometry = ""
        canvas_w = max(int(self.canvas.winfo_width()), 1)
        canvas_h = max(int(self.canvas.winfo_height()), 1)
        if canvas_w > 10 and canvas_h > 10:
            self.canvas.configure(width=canvas_w, height=canvas_h)
        return window_state, geometry, canvas_w, canvas_h

    def _restore_window_layout_guard(self, guard: tuple[str, str, int, int]) -> None:
        window_state, geometry, _canvas_w, _canvas_h = guard

        def restore() -> None:
            try:
                if window_state == "zoomed":
                    self.state("zoomed")
                elif window_state == "normal" and geometry:
                    self.geometry(geometry)
            except tk.TclError:
                return

        self.after_idle(restore)
        self.after(80, restore)

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=10)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(1, weight=1)

        top = ttk.LabelFrame(self, text="Atlas fitting")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        top.columnconfigure(1, weight=1)
        top.columnconfigure(2, weight=1)
        top.columnconfigure(3, weight=1)

        ttk.Label(top, text="1. Input JSON").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.input_json_entry = ttk.Entry(top, textvariable=self.input_json_var)
        self.input_json_entry.grid(row=0, column=1, columnspan=3, sticky="ew", padx=(0, 6), pady=4)
        self.input_json_entry.bind("<Return>", lambda _event: self._load_quicknii_json())
        ttk.Button(top, text="Browse", command=self._browse_input_json, width=8).grid(row=0, column=4, sticky="w", padx=(0, 6), pady=4)

        ttk.Label(top, text="2. Output JSON").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(top, textvariable=self.output_json_var).grid(row=1, column=1, columnspan=3, sticky="ew", padx=(0, 6), pady=4)
        ttk.Button(top, text="Browse", command=self._browse_output_json, width=8).grid(row=1, column=4, sticky="w", padx=(0, 6), pady=4)

        button_row = ttk.Frame(top)
        button_row.grid(row=2, column=0, columnspan=6, sticky="ew", padx=4, pady=(3, 0))
        ttk.Button(button_row, text="Save JSON", command=self._save_quicknii_json, width=16).pack(side="left", padx=4)
        ttk.Button(button_row, text="Restore", command=self._restore_saved_state, width=12).pack(side="left", padx=4)
        self.hover_region_label = tk.Label(
            button_row,
            textvariable=self.hover_region_var,
            font=("Segoe UI", 24, "bold"),
            fg="#082f9a",
            anchor="w",
            justify="left",
            bg=self.cget("bg"),
        )
        self.hover_region_label.pack(side="left", fill="x", expand=True, padx=(16, 4))

        self.canvas = tk.Canvas(self, background="#14171c", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=(4, 2), pady=(0, 4))

        side = ttk.Frame(self, width=self.SIDE_PANEL_WIDTH)
        side.grid(row=1, column=1, sticky="nsew", padx=(2, 4), pady=(0, 4))
        side.grid_propagate(False)
        side.columnconfigure(1, weight=1)
        side.rowconfigure(3, weight=0)

        nav = ttk.LabelFrame(side, text="Slice")
        nav.grid(row=0, column=0, sticky="ew")
        nav.columnconfigure(1, weight=1)
        ttk.Button(nav, text="Prev", command=lambda: self._step_slice(-1), width=10).grid(row=0, column=0, padx=6, pady=6)
        self.slice_combo = ttk.Combobox(nav, state="readonly", height=10, width=44)
        self.slice_combo.grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        self.slice_combo.bind("<<ComboboxSelected>>", self._select_slice_from_combo)
        ttk.Button(nav, text="Next", command=lambda: self._step_slice(1), width=10).grid(row=0, column=2, padx=6, pady=6)
        ttk.Label(
            nav,
            textvariable=self.slice_label_var,
            foreground="#4c5a6a",
            wraplength=self.SIDE_PANEL_WIDTH - 28,
        ).grid(row=1, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 6))
        ttk.Label(nav, text="Display CH").grid(row=2, column=0, sticky="w", padx=8, pady=(0, 6))
        self.display_channel_combo = ttk.Combobox(
            nav,
            state="readonly",
            values=["Auto(JSON)", "CH1", "CH2", "CH3", "CH4"],
            textvariable=self.display_channel_var,
            width=12,
        )
        self.display_channel_combo.grid(row=2, column=1, sticky="w", padx=6, pady=(0, 6))
        self.display_channel_combo.bind("<<ComboboxSelected>>", self._on_display_channel_changed)

        controls = ttk.LabelFrame(side, text="Transform")
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        controls.columnconfigure(2, weight=1)
        controls.columnconfigure(5, weight=1)

        control_specs = [
            ("AP (Bregma um)", self.ap_var, 0, 0, "ap"),
            ("Center ML", self.center_ml_var, 1, 0, "center_ml"),
            ("Center DV", self.center_dv_var, 1, 3, "center_dv"),
            ("Atlas size ML (%)", self.span_ml_var, 2, 0, "size"),
            ("Atlas size DV (%)", self.span_dv_var, 2, 3, "size"),
            ("Tilt ML deg", self.tilt_ml_var, 3, 0, "angle"),
            ("Tilt DV deg", self.tilt_dv_var, 3, 3, "angle"),
            ("Roll deg", self.roll_var, 4, 0, "angle"),
        ]
        for label, variable, row, column, kind in control_specs:
            ttk.Label(controls, text=label).grid(row=row, column=column, sticky="w", padx=6, pady=4)
            entry_frame = ttk.Frame(controls)
            entry_frame.grid(row=row, column=column + 1, sticky="ew", padx=(0, 6), pady=4)
            entry_frame.columnconfigure(1, weight=1)
            if kind == "center_ml":
                minus_label, plus_label = "\u2190", "\u2192"
            elif kind == "center_dv":
                minus_label, plus_label = "\u2191", "\u2193"
            elif kind in {"span", "size"}:
                minus_label, plus_label = "-", "+"
            elif kind == "ap":
                minus_label, plus_label = "-", "+"
            else:
                minus_label, plus_label = "<", ">"
            ttk.Button(
                entry_frame,
                text=minus_label,
                width=2,
                command=lambda name=label, mode=kind: self._nudge_transform(name, mode, -1),
            ).grid(row=0, column=0, padx=(0, 2))
            entry = ttk.Entry(entry_frame, textvariable=variable, width=11)
            entry.grid(row=0, column=1, sticky="ew")
            ttk.Button(
                entry_frame,
                text=plus_label,
                width=2,
                command=lambda name=label, mode=kind: self._nudge_transform(name, mode, 1),
            ).grid(row=0, column=2, padx=(2, 0))
            if kind == "ap":
                ttk.Button(
                    entry_frame,
                    text="-50",
                    width=4,
                    command=lambda: self._adjust_ap(-50.0),
                ).grid(row=0, column=3, padx=(6, 2))
                ttk.Button(
                    entry_frame,
                    text="+50",
                    width=4,
                    command=lambda: self._adjust_ap(50.0),
                ).grid(row=0, column=4, padx=(2, 0))
            entry.bind("<Return>", self._on_transform_commit)
            entry.bind("<FocusOut>", self._on_transform_commit)

        opacity = ttk.LabelFrame(side, text="Display")
        opacity.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        opacity.columnconfigure(1, weight=1)
        ttk.Label(opacity, text="Atlas opacity").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Scale(opacity, from_=0.0, to=1.0, variable=self.opacity_var, command=lambda _v: self._schedule_render()).grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=6)
        ttk.Button(opacity, text="Fit", command=self._reset_view_fit, width=8).grid(row=0, column=2, sticky="e", padx=(0, 6), pady=6)
        marker_tools = ttk.Frame(opacity)
        marker_tools.grid(row=1, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 4))
        ttk.Label(marker_tools, text="Mouse tool").pack(side="left")
        ttk.Radiobutton(marker_tools, text="Pan", value="pan", variable=self.marker_mode_var).pack(side="left", padx=(8, 4))
        ttk.Radiobutton(marker_tools, text="Marker", value="marker", variable=self.marker_mode_var).pack(side="left", padx=4)
        ttk.Radiobutton(marker_tools, text="Omit", value="omit", variable=self.marker_mode_var).pack(side="left", padx=4)
        ttk.Button(marker_tools, text="Clear omit", command=self._clear_current_omit, width=12).pack(side="right", padx=(4, 0))
        ttk.Button(marker_tools, text="Undo omit", command=self._undo_last_omit, width=12).pack(side="right", padx=(4, 0))
        ttk.Button(marker_tools, text="Clear markers", command=self._clear_current_markers, width=14).pack(side="right")
        omit_brush = ttk.Frame(opacity)
        omit_brush.grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 4))
        ttk.Label(omit_brush, text="Omit brush px").pack(side="left")
        ttk.Spinbox(omit_brush, from_=10, to=600, increment=50, textvariable=self.omit_brush_px_var, width=8).pack(side="left", padx=(8, 0))
        ttk.Label(omit_brush, text="Mode").pack(side="left", padx=(14, 4))
        ttk.Radiobutton(omit_brush, text="Brush", value="brush", variable=self.omit_draw_mode_var).pack(side="left", padx=(0, 4))
        ttk.Radiobutton(omit_brush, text="Polygon", value="polygon", variable=self.omit_draw_mode_var).pack(side="left", padx=4)
        ttk.Label(opacity, textvariable=self.preview_mode_var, foreground="#a15b00", wraplength=430, justify="left").grid(row=3, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 2))
        ttk.Label(opacity, textvariable=self.preview_note_var, foreground="#4c5a6a", wraplength=430, justify="left").grid(row=4, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 6))

        log_frame = ttk.LabelFrame(side, text="Log")
        log_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = ScrolledText(log_frame, wrap="word", font=("Consolas", 9), height=8)
        self.log_text.grid(row=0, column=0, sticky="nsew")

        status = ttk.Frame(side)
        status.grid(row=4, column=0, sticky="ew", pady=(8, 0))
        status.columnconfigure(0, weight=1)
        ttk.Label(status, textvariable=self.status_var, anchor="w").grid(row=0, column=0, sticky="ew")
        ttk.Label(status, text=f"Version {APP_VERSION}", foreground="#5f6b7a").grid(row=0, column=1, sticky="e", padx=(8, 0))
        self._suppress_button_space_activation()

    def _bind_shortcuts(self) -> None:
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_left_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_left_release)
        self.canvas.bind("<ButtonPress-3>", self._on_canvas_right_press)
        self.canvas.bind("<B3-Motion>", self._on_canvas_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_canvas_right_release)
        self.canvas.bind("<MouseWheel>", self._on_canvas_mousewheel)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
        self.bind("<Delete>", self._delete_selected_marker)
        self.bind("<BackSpace>", self._delete_selected_marker)
        self.bind("<KeyPress-space>", self._on_space_press)
        self.bind("<KeyRelease-space>", self._on_space_release)
        self.bind("<FocusOut>", self._on_space_focus_out)
        self.bind("<Left>", lambda _e: self._step_slice(-1))
        self.bind("<Right>", lambda _e: self._step_slice(1))
        self.bind("<Prior>", lambda _e: self._adjust_ap(-10.0))
        self.bind("<Next>", lambda _e: self._adjust_ap(10.0))
        self.bind("<KeyPress-equal>", lambda _e: self._apply_zoom(1.0 / 1.08))
        self.bind("<KeyPress-minus>", lambda _e: self._apply_zoom(1.08))

    def _suppress_button_space_activation(self, root: tk.Misc | None = None) -> None:
        """Reserve Space for atlas opacity even when a button still has focus."""

        root = self if root is None else root
        for child in root.winfo_children():
            try:
                widget_class = str(child.winfo_class())
            except tk.TclError:
                continue
            if widget_class in {"Button", "TButton"}:
                try:
                    child.configure(takefocus=False)
                except tk.TclError:
                    pass
                child.bind("<KeyPress-space>", self._on_space_press)
                child.bind("<KeyRelease-space>", self._on_space_release)
            self._suppress_button_space_activation(child)

    def _omit_masks_dir(self, json_path: Path) -> Path:
        return json_path.with_name(f"{json_path.stem}{self.OMIT_MASK_DIR_SUFFIX}")

    def _omit_state_path(self, json_path: Path) -> Path:
        return json_path.with_name(f"{json_path.stem}{self.OMIT_STATE_SUFFIX}")

    def _browse_history_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / self.BROWSE_HISTORY_FILE

    def _load_browse_history(self) -> dict[str, str]:
        path = self._browse_history_path()
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        history: dict[str, str] = {}
        for key, value in payload.items():
            if isinstance(key, str) and isinstance(value, str):
                history[key] = value
        return history

    def _save_browse_history(self) -> None:
        path = self._browse_history_path()
        try:
            self._atomic_write_json(path, self._browse_history)
        except OSError:
            return

    def _atomic_write_json(self, path: Path, payload: object) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=str(path.parent),
                prefix=f"{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except Exception:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise

    def _browse_dir_candidate(self, value: str | os.PathLike[str] | None) -> str | None:
        if not value:
            return None
        try:
            path = Path(value)
        except TypeError:
            return None
        if path.is_file():
            path = path.parent
        if path.exists() and path.is_dir():
            return str(path)
        return None

    def _initial_browse_dir(self, key: str, *candidates: str | os.PathLike[str] | None) -> str:
        remembered = self._browse_dir_candidate(self._browse_history.get(key))
        if remembered:
            return remembered
        for candidate in candidates:
            resolved = self._browse_dir_candidate(candidate)
            if resolved:
                return resolved
        return os.getcwd()

    def _remember_browse_dir(self, key: str, selected_path: str | os.PathLike[str]) -> None:
        path = Path(selected_path)
        directory = path.parent if path.suffix else path
        if directory.exists() and directory.is_dir():
            self._browse_history[key] = str(directory)
            self._save_browse_history()

    def _browse_input_json(self) -> None:
        selected = filedialog.askopenfilename(
            initialdir=self._initial_browse_dir("input_json", self.input_json_var.get(), self.image_folder_var.get()),
            filetypes=[("JSON", "*.json")],
        )
        if selected:
            self.input_json_var.set(selected)
            self._remember_browse_dir("input_json", selected)
            self.output_json_var.set(str(Path(selected)))
            self.status_var.set("Loading JSON...")
            self.after_idle(self._load_quicknii_json)

    def _browse_output_json(self) -> None:
        selected = filedialog.asksaveasfilename(
            initialdir=self._initial_browse_dir("output_json", self.output_json_var.get(), self.input_json_var.get(), self.image_folder_var.get()),
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if selected:
            self.output_json_var.set(selected)
            self._remember_browse_dir("output_json", selected)

    def _browse_session_path(self) -> None:
        selected = filedialog.asksaveasfilename(
            initialdir=self.session_path_var.get() or self.input_json_var.get() or self.image_folder_var.get() or os.getcwd(),
            defaultextension=".yaml",
            filetypes=[("YAML", "*.yaml *.yml")],
        )
        if selected:
            self.session_path_var.set(selected)

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")

    def _clone_omit_stroke(self, stroke: dict[str, object]) -> dict[str, object]:
        return {
            "mode": str(stroke.get("mode", "brush")),
            "size": float(stroke.get("size", 0.0)),
            "points": [list(point) for point in stroke.get("points", [])],
        }

    def _clone_state(self, state: SliceFitState) -> SliceFitState:
        return SliceFitState(**asdict(state))

    def _clone_state_map(self) -> dict[str, SliceFitState]:
        return {name: self._clone_state(state) for name, state in self.slice_states.items()}

    def _update_saved_snapshot(self) -> None:
        self.saved_slice_states = self._clone_state_map()

    def _current_target_voxel_size_um(self) -> float:
        labels_shape = self.atlas.require_labels().shape
        target_shape = self.current_target_resolution
        ratios = [
            float(labels_shape[index]) / float(max(int(target_shape[index]), 1))
            for index in range(3)
        ]
        return float(self.atlas.voxel_size_um * sum(ratios) / len(ratios))

    def _ap_vox_to_bregma_um(self, ap_vox: float) -> float:
        voxel_um = self._current_target_voxel_size_um()
        ap_allen_um = (float(self.current_target_resolution[1]) - 1.0 - float(ap_vox)) * voxel_um
        return float(self.app_config.atlas.allen_bregma_um[0] - ap_allen_um)

    def _ap_bregma_um_to_vox(self, ap_um: float) -> float:
        voxel_um = self._current_target_voxel_size_um()
        ap_allen_um = float(self.app_config.atlas.allen_bregma_um[0] - float(ap_um))
        return float(self.current_target_resolution[1] - 1.0 - (ap_allen_um / voxel_um))

    def _linear_step_vox(self, step_um: float = 20.0) -> float:
        return float(step_um / max(self._current_target_voxel_size_um(), 1e-6))

    def _span_ml_to_ui_size(self, span_ml: float) -> float:
        labels = self.atlas.require_labels()
        return float((labels.shape[0] * 100.0) / max(float(span_ml), 1e-6))

    def _span_dv_to_ui_size(self, span_dv: float) -> float:
        labels = self.atlas.require_labels()
        return float((labels.shape[2] * 100.0) / max(float(span_dv), 1e-6))

    def _ui_size_to_span_ml(self, ui_value: float) -> float:
        labels = self.atlas.require_labels()
        return float((labels.shape[0] * 100.0) / max(float(ui_value), 1e-6))

    def _ui_size_to_span_dv(self, ui_value: float) -> float:
        labels = self.atlas.require_labels()
        return float((labels.shape[2] * 100.0) / max(float(ui_value), 1e-6))

    def _translated_state(self, state: SliceFitState, delta_xyz: tuple[float, float, float]) -> SliceFitState:
        origin, u, v = self._vectors_for_state(state)
        delta = np.asarray(delta_xyz, dtype=np.float64)
        new_origin = origin + delta
        return SliceFitState(
            center_ml=float(state.center_ml + delta[0]),
            center_ap=float(state.center_ap + delta[1]),
            center_dv=float(state.center_dv + delta[2]),
            span_ml=float(state.span_ml),
            span_dv=float(state.span_dv),
            tilt_ml_deg=float(state.tilt_ml_deg),
            tilt_dv_deg=float(state.tilt_dv_deg),
            roll_deg=float(state.roll_deg),
            registration_width=state.registration_width,
            registration_height=state.registration_height,
            origin_vec=tuple(float(value) for value in new_origin),
            u_vec=tuple(float(value) for value in u),
            v_vec=tuple(float(value) for value in v),
            markers=[list(marker) for marker in state.markers],
            omit_strokes=[self._clone_omit_stroke(stroke) for stroke in state.omit_strokes],
        )

    def _copy_state_with_markers(self, state: SliceFitState, markers: list[list[float]]) -> SliceFitState:
        origin, u, v = self._vectors_for_state(state)
        return SliceFitState(
            center_ml=float(state.center_ml),
            center_ap=float(state.center_ap),
            center_dv=float(state.center_dv),
            span_ml=float(state.span_ml),
            span_dv=float(state.span_dv),
            tilt_ml_deg=float(state.tilt_ml_deg),
            tilt_dv_deg=float(state.tilt_dv_deg),
            roll_deg=float(state.roll_deg),
            registration_width=state.registration_width,
            registration_height=state.registration_height,
            origin_vec=tuple(float(value) for value in origin),
            u_vec=tuple(float(value) for value in u),
            v_vec=tuple(float(value) for value in v),
            markers=[list(marker) for marker in markers],
            omit_strokes=[self._clone_omit_stroke(stroke) for stroke in state.omit_strokes],
        )

    def _copy_state_with_omit_strokes(self, state: SliceFitState, omit_strokes: list[dict[str, object]]) -> SliceFitState:
        origin, u, v = self._vectors_for_state(state)
        return SliceFitState(
            center_ml=float(state.center_ml),
            center_ap=float(state.center_ap),
            center_dv=float(state.center_dv),
            span_ml=float(state.span_ml),
            span_dv=float(state.span_dv),
            tilt_ml_deg=float(state.tilt_ml_deg),
            tilt_dv_deg=float(state.tilt_dv_deg),
            roll_deg=float(state.roll_deg),
            registration_width=state.registration_width,
            registration_height=state.registration_height,
            origin_vec=tuple(float(value) for value in origin),
            u_vec=tuple(float(value) for value in u),
            v_vec=tuple(float(value) for value in v),
            markers=[list(marker) for marker in state.markers],
            omit_strokes=[self._clone_omit_stroke(stroke) for stroke in omit_strokes],
        )

    def _shift_pressed(self, event: tk.Event) -> bool:
        return bool(int(event.state) & 0x0001)

    def _ctrl_pressed(self, event: tk.Event) -> bool:
        return bool(int(event.state) & 0x0004)

    def _on_canvas_motion(self, event: tk.Event) -> None:
        self._last_canvas_xy = (float(event.x), float(event.y))
        hover = self._hover_region_text(event.x, event.y)
        self.hover_region_var.set(hover)
        self._update_preview_note()
        self._update_omit_cursor_preview(event.x, event.y)

    def _on_canvas_leave(self, _event: tk.Event) -> None:
        self.hover_region_var.set("")
        self._update_omit_cursor_preview(None, None)

    def _list_image_paths(self, folder: Path) -> list[Path]:
        paths: list[Path] = []
        for path in folder.iterdir():
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            lowered = path.name.lower()
            if "simple segmentation" in lowered or "simple_segmentation" in lowered or "segmentation" in lowered:
                continue
            paths.append(path)
        return sorted(paths)

    def _normalize_filename_key(self, name: str) -> str:
        return Path(name).stem.strip().lower().replace(" ", "_")

    def _channel_neutral_filename_key(self, name: str) -> str:
        """Return a slice key that ignores display channel suffixes such as CH2/CH3."""

        normalized = self._normalize_filename_key(name)
        tokens = re.split(r"[_\-]+", normalized)
        filtered = [token for token in tokens if not re.fullmatch(r"ch\d+", token, flags=re.IGNORECASE)]
        return "_".join(token for token in filtered if token)

    def _selected_display_channel(self) -> str:
        value = str(self.display_channel_var.get() or "").strip().upper()
        if not value or value.startswith("AUTO"):
            return ""
        return value if re.fullmatch(r"CH\d+", value) else ""

    def _display_image_path_for(self, json_image_path: Path) -> Path:
        """Return the image to show on screen without changing the JSON slice filename."""

        channel = self._selected_display_channel()
        if not channel:
            return json_image_path
        swapped_name = re.sub(r"(?i)CH\d+", channel, json_image_path.name, count=1)
        if swapped_name == json_image_path.name:
            return json_image_path
        candidate = json_image_path.with_name(swapped_name)
        if candidate.exists():
            return candidate
        lowered = swapped_name.lower()
        try:
            for sibling in json_image_path.parent.iterdir():
                if sibling.is_file() and sibling.name.lower() == lowered:
                    return sibling
        except OSError:
            pass
        return json_image_path

    def _on_display_channel_changed(self, _event: tk.Event | None = None) -> None:
        if not (0 <= self.current_index < len(self.image_paths)):
            return
        state = self._read_state_from_controls()
        if state is not None:
            self.slice_states[self.image_paths[self.current_index].name] = state
        self._current_base_signature = None
        self._load_current_slice()

    def _omit_strokes_for_image(
        self,
        omit_by_name: dict[str, list[dict[str, object]]],
        image_name: str,
    ) -> list[dict[str, object]]:
        """Find omit strokes by exact filename first, then by channel-neutral slice key."""

        lookup_keys = [
            image_name,
            self._normalize_filename_key(image_name),
            self._channel_neutral_filename_key(image_name),
        ]
        for key in lookup_keys:
            if key in omit_by_name:
                return [self._clone_omit_stroke(stroke) for stroke in omit_by_name[key]]
        return []

    def _sort_paths_anterior_to_posterior(
        self,
        paths: list[Path],
        states: dict[str, SliceFitState],
    ) -> list[Path]:
        def sort_key(path: Path) -> tuple[float, str]:
            state = states.get(path.name)
            if state is None:
                return (float("inf"), path.name.lower())
            try:
                ap_um = self._ap_vox_to_bregma_um(float(state.center_ap))
            except (TypeError, ValueError, OverflowError):
                ap_um = float("-inf")
            if math.isnan(ap_um):
                ap_um = float("-inf")
            return (-ap_um, path.name.lower())

        return sorted(paths, key=sort_key)

    def _load_image_folder(self, reset_target: bool = True) -> None:
        folder = Path(self.image_folder_var.get().strip())
        if not folder.exists():
            messagebox.showerror("Missing folder", f"Image folder not found:\n{folder}")
            return
        paths = self._list_image_paths(folder)
        if not paths:
            messagebox.showerror("No images", f"No image files found under:\n{folder}")
            return
        self.image_paths = paths
        self.image_sizes = {}
        self.slice_states = {}
        if reset_target:
            self.current_target_name = self.app_config.atlas.name
            self.current_target_resolution = tuple(int(value) for value in self.app_config.atlas.quicknii_resolution_vox)
        for path in self.image_paths:
            array = load_image_array(path)
            height, width = array.shape[:2]
            self.image_sizes[path.name] = (height, width)
            self.slice_states[path.name] = self._default_state(width, height)
        self.slice_combo["values"] = [path.name for path in self.image_paths]
        self.current_index = 0
        self.slice_combo.current(0)
        self._update_saved_snapshot()
        self._append_log(f"Loaded {len(self.image_paths)} images from {folder}")
        self._load_current_slice()

    def _default_state(self, width: int, height: int) -> SliceFitState:
        labels = self.atlas.require_labels()
        ml_extent = float(labels.shape[0])
        dv_extent = float(labels.shape[2])
        image_aspect = float(width) / float(max(height, 1))
        scale = min(ml_extent / max(width, 1), dv_extent / max(height, 1))
        span_ml = float(width) * scale
        span_dv = float(height) * scale
        return self._state_from_display_values(
            center_ml=ml_extent / 2.0,
            center_ap=float(labels.shape[1]) / 2.0,
            center_dv=dv_extent / 2.0,
            span_ml=span_ml,
            span_dv=span_dv,
            registration_width=width,
            registration_height=height,
        )

    def _load_current_slice(self) -> None:
        if not (0 <= self.current_index < len(self.image_paths)):
            return
        layout_guard = self._capture_window_layout_guard()
        path = self.image_paths[self.current_index]
        display_path = self._display_image_path_for(path)
        image = ensure_rgb(load_image_array(display_path))
        expected_shape = self.image_sizes[path.name]
        if image.shape[:2] != expected_shape:
            self._append_log(
                f"Display channel image size mismatch for {display_path.name}; using JSON image {path.name}"
            )
            display_path = path
            image = ensure_rgb(load_image_array(path))
        self.current_display_image_path = display_path
        self.current_preview_rgb = image
        self.current_preview_shape = None
        self.current_region_map = None
        self._current_boundary_contours = []
        self.view_zoom_factor = 1.0
        self.current_display_offset = (0.0, 0.0)
        self._preview_overlay_shift = (0.0, 0.0)
        self._drag_overlay_base_alpha = None
        self._drag_overlay_base_state = None
        self._clear_preview_contour_items()
        self._reset_view_on_next_render = True
        self._set_preview_mode(False)
        self._selected_marker_index = None
        self._selected_omit_stroke_index = None
        self._marker_input_ready = False
        self.hover_region_var.set("")
        self._prime_display_geometry_for_current_slice(center_view=True)
        label = f"{path.name}  ({self.current_index + 1}/{len(self.image_paths)})"
        if display_path.name != path.name:
            label = f"{label}  | display: {display_path.name}"
        self.slice_label_var.set(label)
        self._push_state_to_controls(self.slice_states[path.name])
        self._update_preview_note()
        self.status_var.set("Pan: left-drag atlas, wheel zoom, hold Space to fade atlas | Marker: right-click add, left-drag move | Omit: Brush left-drag / Polygon left-click add, right-click close")
        self._schedule_render(interactive=True)
        self._schedule_refine_render()
        self._restore_window_layout_guard(layout_guard)

    def _push_state_to_controls(self, state: SliceFitState) -> None:
        self._ignore_var_updates = True
        try:
            self.ap_var.set(f"{self._ap_vox_to_bregma_um(state.center_ap):.1f}")
            self.center_ml_var.set(f"{state.center_ml:.2f}")
            self.center_dv_var.set(f"{state.center_dv:.2f}")
            self.span_ml_var.set(f"{self._span_ml_to_ui_size(state.span_ml):.1f}")
            self.span_dv_var.set(f"{self._span_dv_to_ui_size(state.span_dv):.1f}")
            self.tilt_ml_var.set(f"{state.tilt_ml_deg:.2f}")
            self.tilt_dv_var.set(f"{state.tilt_dv_deg:.2f}")
            self.roll_var.set(f"{state.roll_deg:.2f}")
        finally:
            self._ignore_var_updates = False

    def _controls_match_state(self, state: SliceFitState) -> bool:
        return (
            self.ap_var.get().strip() == f"{self._ap_vox_to_bregma_um(state.center_ap):.1f}"
            and self.center_ml_var.get().strip() == f"{state.center_ml:.2f}"
            and self.center_dv_var.get().strip() == f"{state.center_dv:.2f}"
            and self.span_ml_var.get().strip() == f"{self._span_ml_to_ui_size(state.span_ml):.1f}"
            and self.span_dv_var.get().strip() == f"{self._span_dv_to_ui_size(state.span_dv):.1f}"
            and self.tilt_ml_var.get().strip() == f"{state.tilt_ml_deg:.2f}"
            and self.tilt_dv_var.get().strip() == f"{state.tilt_dv_deg:.2f}"
            and self.roll_var.get().strip() == f"{state.roll_deg:.2f}"
        )

    def _read_state_from_controls(self) -> SliceFitState | None:
        try:
            current_markers: list[list[float]] = []
            current_omit_strokes: list[dict[str, object]] = []
            if 0 <= self.current_index < len(self.image_paths):
                current_state = self.slice_states[self.image_paths[self.current_index].name]
                if self._controls_match_state(current_state):
                    return self._clone_state(current_state)
                current_markers = [list(marker) for marker in current_state.markers]
                current_omit_strokes = [self._clone_omit_stroke(stroke) for stroke in current_state.omit_strokes]
            return self._state_from_display_values(
                center_ml=float(self.center_ml_var.get().strip()),
                center_ap=self._ap_bregma_um_to_vox(float(self.ap_var.get().strip())),
                center_dv=float(self.center_dv_var.get().strip()),
                span_ml=max(1.0, self._ui_size_to_span_ml(float(self.span_ml_var.get().strip()))),
                span_dv=max(1.0, self._ui_size_to_span_dv(float(self.span_dv_var.get().strip()))),
                tilt_ml_deg=float(self.tilt_ml_var.get().strip()),
                tilt_dv_deg=float(self.tilt_dv_var.get().strip()),
                roll_deg=float(self.roll_var.get().strip()),
                markers=current_markers,
                omit_strokes=current_omit_strokes,
                registration_width=current_state.registration_width,
                registration_height=current_state.registration_height,
            )
        except ValueError:
            return None

    def _state_from_display_values(
        self,
        center_ml: float,
        center_ap: float,
        center_dv: float,
        span_ml: float,
        span_dv: float,
        tilt_ml_deg: float = 0.0,
        tilt_dv_deg: float = 0.0,
        roll_deg: float = 0.0,
        markers: list[list[float]] | None = None,
        omit_strokes: list[dict[str, object]] | None = None,
        registration_width: int | None = None,
        registration_height: int | None = None,
    ) -> SliceFitState:
        provisional = SliceFitState(
            center_ml=float(center_ml),
            center_ap=float(center_ap),
            center_dv=float(center_dv),
            span_ml=max(1.0, float(span_ml)),
            span_dv=max(1.0, float(span_dv)),
            tilt_ml_deg=float(tilt_ml_deg),
            tilt_dv_deg=float(tilt_dv_deg),
            roll_deg=float(roll_deg),
            registration_width=None if registration_width is None else int(registration_width),
            registration_height=None if registration_height is None else int(registration_height),
            markers=[list(marker) for marker in (markers or [])],
            omit_strokes=[self._clone_omit_stroke(stroke) for stroke in (omit_strokes or [])],
        )
        origin, u, v = self._vectors_from_display_state(provisional)
        return SliceFitState(
            center_ml=provisional.center_ml,
            center_ap=provisional.center_ap,
            center_dv=provisional.center_dv,
            span_ml=provisional.span_ml,
            span_dv=provisional.span_dv,
            tilt_ml_deg=provisional.tilt_ml_deg,
            tilt_dv_deg=provisional.tilt_dv_deg,
            roll_deg=provisional.roll_deg,
            registration_width=provisional.registration_width,
            registration_height=provisional.registration_height,
            origin_vec=tuple(float(value) for value in origin),
            u_vec=tuple(float(value) for value in u),
            v_vec=tuple(float(value) for value in v),
            markers=[list(marker) for marker in provisional.markers],
            omit_strokes=[self._clone_omit_stroke(stroke) for stroke in provisional.omit_strokes],
        )

    def _vectors_from_display_state(self, state: SliceFitState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rotation = self._rotation_matrix(state)
        u = rotation @ np.array([state.span_ml, 0.0, 0.0], dtype=np.float64)
        v = rotation @ np.array([0.0, 0.0, state.span_dv], dtype=np.float64)
        center = np.array([state.center_ml, state.center_ap, state.center_dv], dtype=np.float64)
        origin = center - (0.5 * u) - (0.5 * v)
        return origin, u, v

    def _vectors_for_state(self, state: SliceFitState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if state.origin_vec is not None and state.u_vec is not None and state.v_vec is not None:
            return (
                np.asarray(state.origin_vec, dtype=np.float64),
                np.asarray(state.u_vec, dtype=np.float64),
                np.asarray(state.v_vec, dtype=np.float64),
            )
        return self._vectors_from_display_state(state)

    def _rolled_state_from_vectors(self, state: SliceFitState, delta_deg: float) -> SliceFitState:
        origin, u, v = self._vectors_for_state(state)
        axis = np.cross(v, u)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-8:
            return self._state_from_display_values(
                center_ml=state.center_ml,
                center_ap=state.center_ap,
                center_dv=state.center_dv,
                span_ml=state.span_ml,
                span_dv=state.span_dv,
                tilt_ml_deg=state.tilt_ml_deg,
                tilt_dv_deg=state.tilt_dv_deg,
                roll_deg=state.roll_deg + float(delta_deg),
                markers=state.markers,
                omit_strokes=state.omit_strokes,
                registration_width=state.registration_width,
                registration_height=state.registration_height,
            )

        center = origin + (0.5 * u) + (0.5 * v)
        rotation = Rotation.from_rotvec((axis / axis_norm) * math.radians(float(delta_deg))).as_matrix()
        rolled_u = rotation @ u
        rolled_v = rotation @ v
        rolled_origin = center - (0.5 * rolled_u) - (0.5 * rolled_v)
        rolled_state = self._state_from_vectors(
            rolled_origin,
            rolled_u,
            rolled_v,
            markers=state.markers,
            omit_strokes=state.omit_strokes,
            registration_width=state.registration_width,
            registration_height=state.registration_height,
        )
        return replace(rolled_state, roll_deg=float(state.roll_deg + float(delta_deg)))

    def _state_from_vectors(
        self,
        origin: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        markers: list[list[float]] | None = None,
        omit_strokes: list[dict[str, object]] | None = None,
        registration_width: int | None = None,
        registration_height: int | None = None,
    ) -> SliceFitState:
        origin = np.asarray(origin, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        center = origin + (0.5 * u) + (0.5 * v)
        span_ml = max(float(np.linalg.norm(u)), 1.0)
        span_dv = max(float(np.linalg.norm(v)), 1.0)
        u_dir = u / span_ml
        v_dir = v / span_dv

        v_orthogonal = v_dir - (float(np.dot(v_dir, u_dir)) * u_dir)
        v_orthogonal_norm = float(np.linalg.norm(v_orthogonal))
        if v_orthogonal_norm <= 1e-8:
            v_orthogonal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if abs(float(np.dot(v_orthogonal, u_dir))) > 0.9:
                v_orthogonal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            v_orthogonal = v_orthogonal - (float(np.dot(v_orthogonal, u_dir)) * u_dir)
            v_orthogonal /= max(float(np.linalg.norm(v_orthogonal)), 1e-8)
        else:
            v_orthogonal /= v_orthogonal_norm

        axis_ap = np.cross(v_orthogonal, u_dir)
        axis_norm = float(np.linalg.norm(axis_ap))
        if axis_norm <= 1e-8:
            angles = np.zeros(3, dtype=np.float64)
        else:
            axis_ap /= axis_norm
            rotation_matrix = np.column_stack([u_dir, axis_ap, v_orthogonal])
            try:
                angles = Rotation.from_matrix(rotation_matrix).as_euler("zxy", degrees=True)
            except ValueError:
                angles = np.zeros(3, dtype=np.float64)
        return SliceFitState(
            center_ml=float(center[0]),
            center_ap=float(center[1]),
            center_dv=float(center[2]),
            span_ml=span_ml,
            span_dv=span_dv,
            tilt_ml_deg=float(angles[0]),
            tilt_dv_deg=float(angles[1]),
            roll_deg=float(angles[2]),
            registration_width=None if registration_width is None else int(registration_width),
            registration_height=None if registration_height is None else int(registration_height),
            origin_vec=tuple(float(value) for value in origin),
            u_vec=tuple(float(value) for value in u),
            v_vec=tuple(float(value) for value in v),
            markers=[list(marker) for marker in (markers or [])],
            omit_strokes=[self._clone_omit_stroke(stroke) for stroke in (omit_strokes or [])],
        )

    def _set_state_for_current_slice(
        self,
        state: SliceFitState,
        schedule: bool = True,
        interactive: bool = False,
        sync_controls: bool = True,
    ) -> None:
        if not (0 <= self.current_index < len(self.image_paths)):
            return
        self.slice_states[self.image_paths[self.current_index].name] = state
        if sync_controls:
            self._push_state_to_controls(state)
        if schedule:
            self._schedule_render(interactive=interactive)

    def _set_preview_mode(self, active: bool, reason: str = "") -> None:
        self._preview_mode_active = bool(active)
        if self._preview_mode_active:
            detail = f": {reason}" if reason else ""
            self.preview_mode_var.set(f"Preview mode{detail}. Use Save JSON to write changes.")
        else:
            self.preview_mode_var.set("")

    def _refresh_atlas_full(self) -> None:
        self._set_preview_mode(False)
        self._schedule_render(interactive=False, delay_ms=1)

    def _on_transform_commit(self, _event: object | None = None) -> None:
        if self._ignore_var_updates or not (0 <= self.current_index < len(self.image_paths)):
            return
        state = self._read_state_from_controls()
        if state is None:
            self.status_var.set("Invalid transform value")
            return
        self.slice_states[self.image_paths[self.current_index].name] = state
        self._set_preview_mode(False)
        self._schedule_render()

    def _on_canvas_left_press(self, event: tk.Event) -> None:
        if not (0 <= self.current_index < len(self.image_paths)):
            return
        if self.marker_mode_var.get() == "omit":
            self._last_canvas_xy = (float(event.x), float(event.y))
            if self.omit_draw_mode_var.get() == "polygon":
                self._add_polygon_omit_point(event.x, event.y)
            else:
                self._begin_omit_stroke(event.x, event.y)
            return
        if self.marker_mode_var.get() == "marker":
            if not self._marker_input_ready:
                self.status_var.set("Wait for atlas render to finish before editing markers")
                return
            self._last_canvas_xy = (float(event.x), float(event.y))
            existing_index = self._find_marker_near_canvas(event.x, event.y, max_distance_px=12.0)
            if existing_index is None:
                self._selected_marker_index = None
                self.status_var.set("Marker mode: right-click to add marker")
                return
            state = self.slice_states.get(self.image_paths[self.current_index].name)
            if state is None:
                return
            self._selected_marker_index = existing_index
            self._drag_mode = "marker_move"
            self._drag_start_xy = (int(event.x), int(event.y))
            self._drag_start_state = self._clone_state(state)
            self._drag_overlay_base_state = self._clone_state(state)
            self._drag_overlay_base_alpha = None if self._current_atlas_alpha is None else self._current_atlas_alpha.copy()
            self.status_var.set("Drag marker with left button")
            return
        state = self.slice_states.get(self.image_paths[self.current_index].name)
        if state is None:
            return
        self._drag_mode = "pan"
        self._drag_start_xy = (int(event.x), int(event.y))
        self._drag_start_state = self._clone_state(state)
        self._drag_overlay_base_state = self._clone_state(state)
        self._drag_overlay_base_alpha = None if self._current_atlas_alpha is None else self._current_atlas_alpha.copy()

    def _on_canvas_left_drag(self, event: tk.Event) -> None:
        if self._drag_mode is None or self._drag_start_xy is None or self._drag_start_state is None:
            return
        self._last_canvas_xy = (float(event.x), float(event.y))
        if self._drag_mode == "omit_draw":
            self._extend_omit_stroke(event.x, event.y)
            self.status_var.set("Painting omit mask")
            return
        if self._drag_mode == "omit_polygon":
            self._update_live_omit_preview()
            self.status_var.set("Polygon omit: left-click add points, right-click to close")
            return
        if self._drag_mode == "marker_move":
            if not (0 <= self.current_index < len(self.image_paths)):
                return
            state = self.slice_states.get(self.image_paths[self.current_index].name)
            target_point = self._full_image_xy_from_canvas(event.x, event.y, allow_outside=True)
            if state is None or target_point is None or self._selected_marker_index is None or self._selected_marker_index >= len(state.markers):
                return
            markers = [list(marker) for marker in state.markers]
            markers[self._selected_marker_index][2] = float(target_point[0])
            markers[self._selected_marker_index][3] = float(target_point[1])
            new_state = self._copy_state_with_markers(state, markers)
            self._set_state_for_current_slice(new_state, schedule=False, interactive=False, sync_controls=False)
            base_state = self._drag_overlay_base_state or self._drag_start_state
            if base_state is not None:
                self._update_marker_preview_only(base_state, new_state)
            self.status_var.set("Dragging marker")
            return
        width = float(max((self.current_preview_shape or (1, 1))[1], 1))
        height = float(max((self.current_preview_shape or (1, 1))[0], 1))
        dx = float(event.x - self._drag_start_xy[0])
        dy = float(event.y - self._drag_start_xy[1])
        base = self._drag_start_state
        new_state = self._translated_state(
            base,
            (
                float((dx / width) * base.span_ml),
                0.0,
                float((dy / height) * base.span_dv),
            ),
        )
        self._set_state_for_current_slice(new_state, schedule=False, interactive=False, sync_controls=False)
        self._update_pan_preview_only(new_state, (dx, dy))
        self.status_var.set("Drag to move atlas")

    def _on_canvas_left_release(self, event: tk.Event) -> None:
        if self._drag_mode == "pan":
            self._preview_overlay_shift = (0.0, 0.0)
            if 0 <= self.current_index < len(self.image_paths):
                current_state = self.slice_states.get(self.image_paths[self.current_index].name)
                if current_state is not None:
                    self._push_state_to_controls(current_state)
            self._set_preview_mode(True, reason="pan transform")
            self.status_var.set("Preview mode active")
        elif self._drag_mode == "marker_move":
            self._preview_overlay_shift = (0.0, 0.0)
            if 0 <= self.current_index < len(self.image_paths):
                current_state = self.slice_states.get(self.image_paths[self.current_index].name)
                if current_state is not None:
                    self._push_state_to_controls(current_state)
            self._set_preview_mode(False)
            self._marker_input_ready = False
            self.status_var.set("Rebuilding atlas after marker drag")
            self._schedule_render(interactive=False, delay_ms=1)
        elif self._drag_mode == "omit_draw":
            self._commit_live_omit_stroke()
        elif self._drag_mode == "omit_polygon":
            self.status_var.set("Polygon omit: left-click add points, right-click to close")
            self._drag_start_xy = None
            self._drag_start_state = None
            self._drag_overlay_base_alpha = None
            self._drag_overlay_base_state = None
            return
        self._drag_mode = None
        self._drag_start_xy = None
        self._drag_start_state = None
        self._drag_overlay_base_alpha = None
        self._drag_overlay_base_state = None

    def _on_canvas_right_press(self, event: tk.Event) -> None:
        if not (0 <= self.current_index < len(self.image_paths)):
            return
        if self.marker_mode_var.get() == "omit" and self.omit_draw_mode_var.get() == "polygon":
            self._last_canvas_xy = (float(event.x), float(event.y))
            self._complete_polygon_omit()
            return
        if self.marker_mode_var.get() != "marker":
            self.status_var.set("Switch to Marker mode to place markers")
            return
        if not self._marker_input_ready:
            self.status_var.set("Wait for atlas render to finish before editing markers")
            return
        self._last_canvas_xy = (float(event.x), float(event.y))
        existing_index = self._find_marker_near_canvas(event.x, event.y, max_distance_px=10.0)
        if existing_index is not None:
            self._selected_marker_index = existing_index
            self.status_var.set("Marker selected")
            return
        self._add_marker_at_canvas(event.x, event.y)

    def _on_canvas_right_drag(self, event: tk.Event) -> None:
        return

    def _on_canvas_right_release(self, _event: tk.Event) -> None:
        return

    def _add_marker_at_cursor(self, _event: object | None = None) -> None:
        if self.marker_mode_var.get() != "marker":
            self.status_var.set("Switch to Marker mode to add markers")
            return
        if not self._marker_input_ready:
            self.status_var.set("Wait for atlas render to finish before editing markers")
            return
        if self._last_canvas_xy is None:
            self.status_var.set("Move cursor over image first")
            return
        self._add_marker_at_canvas(self._last_canvas_xy[0], self._last_canvas_xy[1])

    def _event_is_text_input(self, event: object | None) -> bool:
        widget = getattr(event, "widget", None)
        if widget is None:
            return False
        try:
            widget_class = str(widget.winfo_class())
        except tk.TclError:
            return False
        return widget_class in {"Entry", "TEntry", "Spinbox", "TSpinbox", "Text", "TCombobox"}

    def _effective_atlas_opacity(self) -> float:
        opacity = float(self.opacity_var.get())
        if self._space_opacity_active:
            return max(0.05, opacity * float(self.space_opacity_factor))
        return opacity

    def _on_space_press(self, event: tk.Event) -> str | None:
        if self._event_is_text_input(event):
            return None
        if self._space_release_after_id is not None:
            self.after_cancel(self._space_release_after_id)
            self._space_release_after_id = None
        if self._space_opacity_active:
            return "break"
        self._space_opacity_active = True
        self._refresh_atlas_opacity_only()
        return "break"

    def _on_space_release(self, event: tk.Event) -> str | None:
        if self._event_is_text_input(event):
            return None
        if self._space_release_after_id is not None:
            self.after_cancel(self._space_release_after_id)
        self._space_release_after_id = self.after(90, self._finish_space_release)
        return "break"

    def _finish_space_release(self) -> None:
        self._space_release_after_id = None
        if not self._space_opacity_active:
            return
        self._space_opacity_active = False
        self._refresh_atlas_opacity_only()

    def _on_space_focus_out(self, _event: tk.Event) -> None:
        if self._space_release_after_id is not None:
            self.after_cancel(self._space_release_after_id)
            self._space_release_after_id = None
        if self._space_opacity_active:
            self._space_opacity_active = False
            self._refresh_atlas_opacity_only()

    def _refresh_atlas_opacity_only(self) -> None:
        if self.current_region_map is None or self.current_composite_shape is None:
            return
        atlas_overlay = self._compose_atlas_overlay(
            self.current_region_map,
            self._effective_atlas_opacity(),
            (self.current_composite_shape[1], self.current_composite_shape[0]),
        )
        self._current_atlas_alpha = np.asarray(atlas_overlay.getchannel("A"), dtype=np.uint8)
        self._cache_boundary_preview_geometry(self._current_atlas_alpha)
        if (
            self.current_atlas_photo is None
            or self.current_atlas_photo.width() != atlas_overlay.size[0]
            or self.current_atlas_photo.height() != atlas_overlay.size[1]
        ):
            self.current_atlas_photo = ImageTk.PhotoImage(atlas_overlay)
        else:
            self.current_atlas_photo.paste(atlas_overlay)
        if self._canvas_atlas_item is None:
            self._canvas_atlas_item = self.canvas.create_image(
                self.current_display_offset[0],
                self.current_display_offset[1],
                anchor="nw",
                image=self.current_atlas_photo,
            )
        else:
            self.canvas.itemconfigure(self._canvas_atlas_item, image=self.current_atlas_photo)
            self.canvas.coords(
                self._canvas_atlas_item,
                self.current_display_offset[0],
                self.current_display_offset[1],
            )
        self._set_atlas_overlay_visibility(True)

    def _begin_omit_stroke(self, x: float, y: float) -> None:
        state = self._read_state_from_controls()
        point = self._full_image_xy_from_canvas(x, y)
        if state is None or point is None:
            return
        self._clear_live_omit_preview()
        self._live_omit_mode = "brush"
        self._live_omit_size_px = float(max(float(self.omit_brush_px_var.get()), 1.0))
        self._live_omit_points = [[float(point[0]), float(point[1])]]
        self._selected_omit_stroke_index = None
        self._drag_mode = "omit_draw"
        self._drag_start_xy = (int(x), int(y))
        self._drag_start_state = self._clone_state(state)
        self._update_live_omit_preview()
        self.status_var.set("Painting omit mask")

    def _add_polygon_omit_point(self, x: float, y: float) -> None:
        state = self._read_state_from_controls()
        point = self._full_image_xy_from_canvas(x, y)
        if state is None or point is None:
            return
        if self._drag_mode != "omit_polygon" or not self._live_omit_points:
            self._clear_live_omit_preview()
            self._live_omit_mode = "polygon"
            self._live_omit_size_px = float(max(float(self.omit_brush_px_var.get()), 1.0))
            self._live_omit_points = []
            self._selected_omit_stroke_index = None
            self._drag_mode = "omit_polygon"
            self._drag_start_xy = (int(x), int(y))
            self._drag_start_state = self._clone_state(state)
        if self._live_omit_points:
            last = self._live_omit_points[-1]
            if math.hypot(float(point[0]) - float(last[0]), float(point[1]) - float(last[1])) < 1.5:
                return
        self._live_omit_points.append([float(point[0]), float(point[1])])
        self._update_live_omit_preview()
        if len(self._live_omit_points) >= 3:
            self.status_var.set("Polygon omit: left-click add points, right-click to close")
        else:
            self.status_var.set("Polygon omit: add at least 3 points")

    def _extend_omit_stroke(self, x: float, y: float) -> None:
        if not (0 <= self.current_index < len(self.image_paths)) or not self._live_omit_points:
            return
        point = self._full_image_xy_from_canvas(x, y)
        if point is None:
            return
        points = self._live_omit_points
        if points:
            last = points[-1]
            if math.hypot(float(point[0]) - float(last[0]), float(point[1]) - float(last[1])) < 1.5:
                return
        points.append([float(point[0]), float(point[1])])
        self._update_live_omit_preview()

    def _clear_current_omit(self) -> None:
        if not (0 <= self.current_index < len(self.image_paths)):
            return
        self._clear_live_omit_preview()
        self._live_omit_points = []
        self._live_omit_mode = "brush"
        if self._drag_mode == "omit_polygon":
            self._drag_mode = None
        name = self.image_paths[self.current_index].name
        state = self.slice_states[name]
        self._selected_omit_stroke_index = None
        updated_state = self._copy_state_with_omit_strokes(state, [])
        self._set_state_for_current_slice(updated_state, schedule=False)
        if not self._refresh_omit_overlay_only(updated_state):
            self._schedule_render(interactive=False, delay_ms=1)
        self.status_var.set("Cleared omit mask")
        self._append_log(f"Cleared omit mask on {name}")

    def _undo_last_omit(self) -> None:
        if not (0 <= self.current_index < len(self.image_paths)):
            return
        if self._live_omit_points:
            if self._live_omit_mode == "polygon" and len(self._live_omit_points) > 1:
                self._live_omit_points.pop()
                self._update_live_omit_preview()
                self.status_var.set("Removed last polygon point")
                return
            self._clear_live_omit_preview()
            self._live_omit_points = []
            self._live_omit_mode = "brush"
            if self._drag_mode == "omit_polygon":
                self._drag_mode = None
            self.status_var.set("Canceled current omit stroke")
            return
        name = self.image_paths[self.current_index].name
        state = self.slice_states[name]
        if not state.omit_strokes:
            self.status_var.set("No omit stroke to undo")
            return
        omit_strokes = [self._clone_omit_stroke(stroke) for stroke in state.omit_strokes[:-1]]
        self._selected_omit_stroke_index = None
        updated_state = self._copy_state_with_omit_strokes(state, omit_strokes)
        self._set_state_for_current_slice(updated_state, schedule=False)
        if not self._refresh_omit_overlay_only(updated_state):
            self._schedule_render(interactive=False, delay_ms=1)
        self.status_var.set("Undid last omit stroke")
        self._append_log(f"Undid last omit stroke on {name}")

    def _add_marker_at_canvas(self, x: float, y: float) -> None:
        if not (0 <= self.current_index < len(self.image_paths)):
            return
        if not self._marker_input_ready:
            self.status_var.set("Wait for atlas render to finish before editing markers")
            return
        state = self.slice_states.get(self.image_paths[self.current_index].name)
        if state is None:
            return
        base_state = self._clone_state(state)
        source_point = self._registration_source_from_canvas(x, y, state, allow_outside=True)
        if source_point is None:
            return
        target_point = self._full_image_xy_from_canvas(x, y, allow_outside=True)
        if target_point is None:
            return
        markers = [list(marker) for marker in state.markers]
        markers.append([float(source_point[0]), float(source_point[1]), float(target_point[0]), float(target_point[1])])
        self._selected_marker_index = len(markers) - 1
        new_state = self._copy_state_with_markers(state, markers)
        self._set_state_for_current_slice(new_state, schedule=False, interactive=False, sync_controls=True)
        self._set_preview_mode(True, reason="marker edit")
        if self._current_atlas_alpha is not None:
            self._drag_overlay_base_alpha = self._current_atlas_alpha.copy()
            self._drag_overlay_base_state = base_state
            self._update_marker_preview_only(base_state, new_state)
        else:
            self._draw_marker_canvas_items(new_state)
        self._update_preview_note()
        self.status_var.set("Marker added (preview mode)")
        self._append_log(f"Added marker on {self.image_paths[self.current_index].name}")

    def _clear_live_omit_preview(self) -> None:
        if self._live_omit_canvas_item is not None:
            self.canvas.delete(self._live_omit_canvas_item)
            self._live_omit_canvas_item = None

    def _update_live_omit_preview(self) -> None:
        self._clear_live_omit_preview()
        if not self._live_omit_points:
            return
        scaled_points = [
            self._canvas_xy_from_full_image(float(point[0]), float(point[1]))
            for point in self._live_omit_points
        ]
        if self._live_omit_mode == "polygon":
            if len(scaled_points) == 1:
                x, y = scaled_points[0]
                self._live_omit_canvas_item = self.canvas.create_line(
                    x - 5,
                    y,
                    x + 5,
                    y,
                    x,
                    y - 5,
                    x,
                    y + 5,
                    fill="#ff5050",
                    width=2,
                )
                return
            preview_points = list(scaled_points)
            if self._last_canvas_xy is not None:
                preview_points.append((float(self._last_canvas_xy[0]), float(self._last_canvas_xy[1])))
            coords = [value for point in preview_points for value in point]
            if len(preview_points) >= 3:
                self._live_omit_canvas_item = self.canvas.create_polygon(
                    *coords,
                    fill="",
                    outline="#ff5050",
                    width=2,
                )
            else:
                self._live_omit_canvas_item = self.canvas.create_line(
                    *coords,
                    fill="#ff5050",
                    width=2,
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                    smooth=False,
                )
            return
        width_px = max(1, int(round(float(self._live_omit_size_px) * float(self.current_preview_scale))))
        if len(scaled_points) == 1:
            x, y = scaled_points[0]
            radius = max(1.0, width_px * 0.5)
            self._live_omit_canvas_item = self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#ff5050",
                outline="",
            )
        else:
            coords = [value for point in scaled_points for value in point]
            self._live_omit_canvas_item = self.canvas.create_line(
                *coords,
                fill="#ff5050",
                width=width_px,
                capstyle=tk.ROUND,
                joinstyle=tk.ROUND,
                smooth=False,
            )

    def _complete_polygon_omit(self) -> None:
        if self._live_omit_mode != "polygon" or not self._live_omit_points:
            return
        if len(self._live_omit_points) < 3:
            self.status_var.set("Polygon omit needs at least 3 points")
            return
        self._commit_live_omit_stroke(mode="polygon")
        self._drag_mode = None
        self._drag_start_xy = None
        self._drag_start_state = None

    def _commit_live_omit_stroke(self, mode: str | None = None) -> None:
        if not (0 <= self.current_index < len(self.image_paths)):
            self._clear_live_omit_preview()
            self._live_omit_points = []
            self._live_omit_mode = "brush"
            return
        state = self._read_state_from_controls()
        if state is None or not self._live_omit_points:
            self._clear_live_omit_preview()
            self._live_omit_points = []
            self._live_omit_mode = "brush"
            return
        omit_strokes = [self._clone_omit_stroke(stroke) for stroke in state.omit_strokes]
        stroke_mode = str(mode or self._live_omit_mode or "brush")
        omit_strokes.append(
            {
                "mode": stroke_mode,
                "size": float(max(self._live_omit_size_px, 1.0)),
                "points": [[float(point[0]), float(point[1])] for point in self._live_omit_points],
            }
        )
        self._clear_live_omit_preview()
        self._live_omit_points = []
        self._live_omit_mode = "brush"
        self._selected_omit_stroke_index = len(omit_strokes) - 1
        updated_state = self._copy_state_with_omit_strokes(state, omit_strokes)
        self._set_state_for_current_slice(updated_state, schedule=False)
        if self._refresh_omit_overlay_only(updated_state):
            self.status_var.set("Updated omit mask")
        else:
            self.status_var.set("Updating omit mask")
            self._schedule_render(interactive=False, delay_ms=1)

    def _refresh_omit_overlay_only(self, state: SliceFitState) -> bool:
        """Refresh only the red omit overlay layer without rebuilding the atlas map."""

        if self.current_preview_rgb is None:
            return False
        composite_shape = self.current_composite_shape or self.current_preview_shape
        if composite_shape is None:
            return False
        image_size = (int(composite_shape[1]), int(composite_shape[0]))
        omit_overlay = self._compose_omit_overlay(state.omit_strokes, image_size)
        if (
            self.current_omit_photo is None
            or self.current_omit_photo.width() != omit_overlay.size[0]
            or self.current_omit_photo.height() != omit_overlay.size[1]
        ):
            self.current_omit_photo = ImageTk.PhotoImage(omit_overlay)
        else:
            self.current_omit_photo.paste(omit_overlay)
        if self._canvas_omit_item is None:
            self._canvas_omit_item = self.canvas.create_image(
                self.current_display_offset[0],
                self.current_display_offset[1],
                anchor="nw",
                image=self.current_omit_photo,
            )
        else:
            self.canvas.itemconfigure(self._canvas_omit_item, image=self.current_omit_photo)
            self.canvas.coords(
                self._canvas_omit_item,
                self.current_display_offset[0],
                self.current_display_offset[1],
            )
        self._draw_marker_canvas_items(state)
        if self._last_canvas_xy is not None:
            self._update_omit_cursor_preview(self._last_canvas_xy[0], self._last_canvas_xy[1])
        self._update_preview_note()
        return True

    def _on_canvas_mousewheel(self, event: tk.Event) -> None:
        if not (0 <= self.current_index < len(self.image_paths)):
            return
        self._last_canvas_xy = (float(event.x), float(event.y))
        step = -1.0 if float(event.delta) > 0 else 1.0
        if self._ctrl_pressed(event):
            self._adjust_roll(step * 1.5)
        elif self._shift_pressed(event):
            self._adjust_ap(step * 10.0)
        else:
            factor = 1.08 if step < 0 else 1.0 / 1.08
            self._apply_zoom(factor)

    def _full_image_xy_from_canvas(self, x: float, y: float, allow_outside: bool = False) -> tuple[float, float] | None:
        if self.current_preview_rgb is None or self.current_preview_scale <= 0.0:
            return None
        offset_x, offset_y = self.current_display_offset
        composite_x = float(x) - offset_x
        composite_y = float(y) - offset_y
        image_offset_x, image_offset_y = self.current_image_offset_in_composite
        if allow_outside:
            composite_h, composite_w = self.current_composite_shape or self.current_preview_shape or self.current_preview_rgb.shape[:2]
            if (
                composite_x < 0.0
                or composite_y < 0.0
                or composite_x > max(float(composite_w) - 1.0, 0.0)
                or composite_y > max(float(composite_h) - 1.0, 0.0)
            ):
                return None
        else:
            display_h, display_w = self.current_preview_shape or self.current_preview_rgb.shape[:2]
            image_x = composite_x - image_offset_x
            image_y = composite_y - image_offset_y
            if image_x < 0.0 or image_y < 0.0 or image_x > max(float(display_w) - 1.0, 0.0) or image_y > max(float(display_h) - 1.0, 0.0):
                return None
            return image_x / self.current_preview_scale, image_y / self.current_preview_scale
        image_x = composite_x - image_offset_x
        image_y = composite_y - image_offset_y
        return image_x / self.current_preview_scale, image_y / self.current_preview_scale

    def _canvas_xy_from_full_image(self, x: float, y: float) -> tuple[float, float]:
        return (
            float(self.current_display_offset[0]) + float(self.current_image_offset_in_composite[0]) + (float(x) * float(self.current_preview_scale)),
            float(self.current_display_offset[1]) + float(self.current_image_offset_in_composite[1]) + (float(y) * float(self.current_preview_scale)),
        )

    def _update_omit_cursor_preview(self, canvas_x: float | None, canvas_y: float | None) -> None:
        if self._omit_preview_item is not None:
            self.canvas.delete(self._omit_preview_item)
            self._omit_preview_item = None
        if self.marker_mode_var.get() != "omit" or canvas_x is None or canvas_y is None:
            return
        if self.omit_draw_mode_var.get() == "polygon":
            return
        point = self._full_image_xy_from_canvas(canvas_x, canvas_y)
        if point is None:
            return
        center_x, center_y = self._canvas_xy_from_full_image(point[0], point[1])
        radius = max(2.0, 0.5 * float(self.omit_brush_px_var.get()) * float(self.current_preview_scale))
        self._omit_preview_item = self.canvas.create_oval(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            outline="#ff5050",
            width=2,
            dash=(4, 3),
        )

    def _registration_source_from_canvas(self, x: float, y: float, state: SliceFitState, allow_outside: bool = False) -> np.ndarray | None:
        full_xy = self._full_image_xy_from_canvas(x, y, allow_outside=allow_outside)
        if full_xy is None or not (0 <= self.current_index < len(self.image_paths)):
            return None
        image_path = self.image_paths[self.current_index]
        image_shape = self.image_sizes[image_path.name]
        registration_slice = self._registration_slice_for(image_path, state, use_preview=False)
        mapper = build_marker_inverse_warp(registration_slice, image_shape)
        registration_xy = np.asarray(
            [
                float(full_xy[0]) * (float(registration_slice.width) / float(max(image_shape[1], 1))),
                float(full_xy[1]) * (float(registration_slice.height) / float(max(image_shape[0], 1))),
            ],
            dtype=np.float64,
        )
        if mapper is None:
            return registration_xy
        mapped = mapper.transform_point(registration_xy)
        if mapped is None or not np.all(np.isfinite(mapped)):
            self.status_var.set("Marker must be placed inside the current warped atlas region")
            return None
        return mapped

    def _hover_region_text(self, x: float, y: float) -> str:
        if self.current_region_map is None:
            return ""
        offset_x, offset_y = self.current_display_offset
        shift_x, shift_y = self._preview_overlay_shift
        rx = int(round(float(x) - offset_x))
        ry = int(round(float(y) - offset_y))
        rx = int(round(rx - shift_x))
        ry = int(round(ry - shift_y))
        if ry < 0 or rx < 0 or ry >= self.current_region_map.shape[0] or rx >= self.current_region_map.shape[1]:
            return ""
        region_id = int(self.current_region_map[ry, rx])
        if region_id <= 0:
            return "Outside atlas"
        region = self.atlas.region_for_id(region_id)
        if region is None:
            return f"Region {region_id}"
        return f"{region.name} ({region.region_id})"

    def _marker_target_canvas_points(self, state: SliceFitState) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        offset_x, offset_y = self.current_display_offset
        image_offset_x, image_offset_y = self.current_image_offset_in_composite
        for marker in state.markers:
            if len(marker) < 4:
                continue
            points.append(
                (
                    offset_x + image_offset_x + (marker[2] * self.current_preview_scale),
                    offset_y + image_offset_y + (marker[3] * self.current_preview_scale),
                )
            )
        return points

    def _marker_target_composite_points(self, state: SliceFitState) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        image_offset_x, image_offset_y = self.current_image_offset_in_composite
        for marker in state.markers:
            if len(marker) < 4:
                continue
            points.append(
                (
                    image_offset_x + (float(marker[2]) * self.current_preview_scale),
                    image_offset_y + (float(marker[3]) * self.current_preview_scale),
                )
            )
        return points

    def _marker_source_canvas_points(self, state: SliceFitState) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        if not (0 <= self.current_index < len(self.image_paths)):
            return points
        image_path = self.image_paths[self.current_index]
        image_height, image_width = self.image_sizes[image_path.name]
        registration_slice = self._registration_slice_for(image_path, state, use_preview=False)
        reg_width = float(max(registration_slice.width, 1))
        reg_height = float(max(registration_slice.height, 1))
        scale_x = float(image_width) / reg_width
        scale_y = float(image_height) / reg_height
        offset_x, offset_y = self.current_display_offset
        image_offset_x, image_offset_y = self.current_image_offset_in_composite
        for marker in state.markers:
            if len(marker) < 4:
                continue
            points.append(
                (
                    offset_x + image_offset_x + (float(marker[0]) * scale_x * self.current_preview_scale),
                    offset_y + image_offset_y + (float(marker[1]) * scale_y * self.current_preview_scale),
                )
            )
        return points

    def _marker_source_composite_points(self, state: SliceFitState) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        if not (0 <= self.current_index < len(self.image_paths)):
            return points
        image_path = self.image_paths[self.current_index]
        image_height, image_width = self.image_sizes[image_path.name]
        registration_slice = self._registration_slice_for(image_path, state, use_preview=False)
        reg_width = float(max(registration_slice.width, 1))
        reg_height = float(max(registration_slice.height, 1))
        scale_x = float(image_width) / reg_width
        scale_y = float(image_height) / reg_height
        image_offset_x, image_offset_y = self.current_image_offset_in_composite
        for marker in state.markers:
            if len(marker) < 4:
                continue
            points.append(
                (
                    image_offset_x + (float(marker[0]) * scale_x * self.current_preview_scale),
                    image_offset_y + (float(marker[1]) * scale_y * self.current_preview_scale),
                )
            )
        return points

    def _cache_boundary_points(self, alpha_array: np.ndarray) -> None:
        alpha_u8 = np.asarray(alpha_array, dtype=np.uint8)
        ys, xs = np.nonzero(alpha_u8 > 0)
        if xs.size == 0:
            self._current_boundary_points = np.zeros((0, 2), dtype=np.float64)
            self._current_boundary_alpha_values = np.zeros((0,), dtype=np.uint8)
            return
        self._current_boundary_points = np.column_stack((xs, ys)).astype(np.float64)
        self._current_boundary_alpha_values = alpha_u8[ys, xs].astype(np.uint8)

    def _cache_boundary_contours(self, alpha_array: np.ndarray) -> None:
        self._current_boundary_contours = []
        if find_contours is None:
            return
        alpha_u8 = np.asarray(alpha_array, dtype=np.uint8)
        if alpha_u8.size == 0 or not np.any(alpha_u8 > 0):
            return
        contour_mask = (alpha_u8 > 0).astype(np.uint8)
        contours = find_contours(contour_mask, 0.5)
        cached: list[np.ndarray] = []
        for contour in contours:
            if contour.shape[0] < 4:
                continue
            if approximate_polygon is not None:
                contour = approximate_polygon(contour, tolerance=0.85)
            if contour.shape[0] < 4:
                continue
            if contour.shape[0] > 1400:
                stride = max(1, int(math.ceil(contour.shape[0] / 1400.0)))
                contour = contour[::stride]
            xy = np.column_stack((contour[:, 1], contour[:, 0])).astype(np.float64)
            if xy.shape[0] >= 4:
                cached.append(xy)
        self._current_boundary_contours = cached

    def _cache_boundary_preview_geometry(self, alpha_array: np.ndarray) -> None:
        self._cache_boundary_points(alpha_array)
        self._cache_boundary_contours(alpha_array)

    def _atlas_overlay_from_alpha(self, alpha_array: np.ndarray) -> Image.Image:
        alpha_u8 = np.clip(np.asarray(alpha_array, dtype=np.float64), 0.0, 255.0).astype(np.uint8)
        overlay = Image.new("RGBA", (alpha_u8.shape[1], alpha_u8.shape[0]), (95, 210, 255, 0))
        overlay.putalpha(Image.fromarray(alpha_u8, mode="L"))
        return overlay

    def _clear_preview_contour_items(self) -> None:
        for item_id in self._preview_contour_canvas_items:
            self.canvas.delete(item_id)
        self._preview_contour_canvas_items.clear()

    def _thin_preview_lines_active(self) -> bool:
        return self._drag_mode == "marker_move" or self._preview_mode_active

    def _set_atlas_overlay_visibility(self, visible: bool) -> None:
        if self._canvas_atlas_item is None:
            return
        self.canvas.itemconfigure(self._canvas_atlas_item, state="normal" if visible else "hidden")

    def _apply_atlas_overlay_alpha(self, alpha_array: np.ndarray, shift_xy: tuple[float, float] = (0.0, 0.0)) -> None:
        self._clear_preview_contour_items()
        self._set_atlas_overlay_visibility(True)
        atlas_overlay = self._atlas_overlay_from_alpha(alpha_array)
        if (
            self.current_atlas_photo is None
            or self.current_atlas_photo.width() != atlas_overlay.size[0]
            or self.current_atlas_photo.height() != atlas_overlay.size[1]
        ):
            self.current_atlas_photo = ImageTk.PhotoImage(atlas_overlay)
        else:
            self.current_atlas_photo.paste(atlas_overlay)
        atlas_x = self.current_display_offset[0] + float(shift_xy[0])
        atlas_y = self.current_display_offset[1] + float(shift_xy[1])
        if self._canvas_atlas_item is None:
            self._canvas_atlas_item = self.canvas.create_image(
                atlas_x,
                atlas_y,
                anchor="nw",
                image=self.current_atlas_photo,
            )
        else:
            self.canvas.itemconfigure(self._canvas_atlas_item, image=self.current_atlas_photo)
            self.canvas.coords(self._canvas_atlas_item, atlas_x, atlas_y)

    def _draw_preview_contours(
        self,
        contour_arrays: list[np.ndarray],
        shift_xy: tuple[float, float] = (0.0, 0.0),
    ) -> bool:
        if not contour_arrays:
            self._clear_preview_contour_items()
            self._set_atlas_overlay_visibility(True)
            return False
        self._set_atlas_overlay_visibility(False)
        atlas_x = float(self.current_display_offset[0] + shift_xy[0])
        atlas_y = float(self.current_display_offset[1] + shift_xy[1])
        line_width = 1
        while len(self._preview_contour_canvas_items) < len(contour_arrays):
            self._preview_contour_canvas_items.append(
                self.canvas.create_line(
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    fill="#5fd2ff",
                    width=line_width,
                    smooth=False,
                    joinstyle=tk.ROUND,
                    capstyle=tk.ROUND,
                )
            )
        for item_id, contour in zip(self._preview_contour_canvas_items, contour_arrays, strict=False):
            if contour.shape[0] < 2:
                self.canvas.itemconfigure(item_id, state="hidden")
                continue
            canvas_points = contour.copy()
            canvas_points[:, 0] += atlas_x
            canvas_points[:, 1] += atlas_y
            coords = canvas_points.reshape(-1).tolist()
            self.canvas.coords(item_id, *coords)
            self.canvas.itemconfigure(item_id, state="normal", fill="#5fd2ff", width=line_width)
        for extra_id in self._preview_contour_canvas_items[len(contour_arrays) :]:
            self.canvas.itemconfigure(extra_id, state="hidden")
        return True

    def _warp_boundary_contours_for_marker_preview(
        self,
        base_alpha: np.ndarray,
        base_state: SliceFitState,
        next_state: SliceFitState,
    ) -> list[np.ndarray] | None:
        image_size = (int(base_alpha.shape[1]), int(base_alpha.shape[0]))
        base_targets = self._marker_target_composite_points(base_state)
        next_targets = self._marker_target_composite_points(next_state)
        if not base_targets or len(base_targets) != len(next_targets):
            return None
        contours = self._current_boundary_contours
        if not contours:
            self._cache_boundary_preview_geometry(base_alpha)
            contours = self._current_boundary_contours
        if not contours:
            return None
        marker_rows = [
            [float(target_pt[0]), float(target_pt[1]), float(source_pt[0]), float(source_pt[1])]
            for source_pt, target_pt in zip(base_targets, next_targets, strict=False)
        ]
        mapper = build_piecewise_affine_mapper(image_size[0], image_size[1], marker_rows)
        if mapper is None:
            return None
        warped_contours: list[np.ndarray] = []
        for contour in contours:
            if contour.shape[0] < 2:
                continue
            warped = mapper.map_target_to_source(contour)
            if warped.shape[0] >= 2:
                warped_contours.append(warped)
        return warped_contours or None

    def _warp_atlas_alpha_for_marker_preview(
        self,
        base_alpha: np.ndarray,
        base_state: SliceFitState,
        next_state: SliceFitState,
    ) -> np.ndarray:
        image_size = (int(base_alpha.shape[1]), int(base_alpha.shape[0]))
        base_targets = self._marker_target_composite_points(base_state)
        next_targets = self._marker_target_composite_points(next_state)
        if not base_targets or len(base_targets) != len(next_targets):
            return base_alpha
        marker_rows = [
            [float(target_pt[0]), float(target_pt[1]), float(source_pt[0]), float(source_pt[1])]
            for source_pt, target_pt in zip(base_targets, next_targets, strict=False)
        ]
        mapper = build_piecewise_affine_mapper(image_size[0], image_size[1], marker_rows)
        if mapper is None:
            return base_alpha
        boundary_points = self._current_boundary_points
        boundary_values = self._current_boundary_alpha_values
        if boundary_points is None or boundary_values is None:
            self._cache_boundary_preview_geometry(base_alpha)
            boundary_points = self._current_boundary_points
            boundary_values = self._current_boundary_alpha_values
        if boundary_points is None or boundary_values is None or boundary_points.size == 0:
            return np.zeros_like(base_alpha, dtype=np.uint8)
        mapped = mapper.map_target_to_source(boundary_points)
        out = np.zeros_like(base_alpha, dtype=np.uint8)
        xi = np.rint(mapped[:, 0]).astype(np.int32)
        yi = np.rint(mapped[:, 1]).astype(np.int32)
        width = int(image_size[0])
        height = int(image_size[1])
        flat = out.ravel()
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            xj = xi + dx
            yj = yi + dy
            inside = (xj >= 0) & (xj < width) & (yj >= 0) & (yj < height)
            if not np.any(inside):
                continue
            np.maximum.at(flat, yj[inside] * width + xj[inside], boundary_values[inside])
        return out

    def _clear_marker_canvas_items(self) -> None:
        for item_id in self._marker_canvas_items:
            self.canvas.delete(item_id)
        self._marker_canvas_items.clear()

    def _draw_marker_canvas_items(
        self,
        state: SliceFitState,
        source_shift: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self._clear_marker_canvas_items()
        if not state.markers:
            return
        source_points = self._marker_source_canvas_points(state)
        target_points = self._marker_target_canvas_points(state)
        shift_x = float(source_shift[0])
        shift_y = float(source_shift[1])
        thin_preview = self._thin_preview_lines_active()
        connector_width = 1 if thin_preview else 4
        outline_width = 2 if thin_preview else 3
        target_radius = 9 if thin_preview else 13
        show_source_helper = True
        for index, marker in enumerate(state.markers):
            if len(marker) < 4:
                continue
            sx, sy = source_points[index] if index < len(source_points) else (
                float(self.current_display_offset[0] + self.current_image_offset_in_composite[0] + (marker[0] * self.current_preview_scale)),
                float(self.current_display_offset[1] + self.current_image_offset_in_composite[1] + (marker[1] * self.current_preview_scale)),
            )
            tx, ty = target_points[index] if index < len(target_points) else (
                float(self.current_display_offset[0] + self.current_image_offset_in_composite[0] + (marker[2] * self.current_preview_scale)),
                float(self.current_display_offset[1] + self.current_image_offset_in_composite[1] + (marker[3] * self.current_preview_scale)),
            )
            sx += shift_x
            sy += shift_y
            color = "#ffdc50" if index == self._selected_marker_index else "#ff6eff"
            source_color = "#78ffff"
            if show_source_helper:
                self._marker_canvas_items.append(self.canvas.create_line(sx, sy, tx, ty, fill=color, width=connector_width))
                self._marker_canvas_items.append(
                    self.canvas.create_oval(
                        sx - 2,
                        sy - 2,
                        sx + 2,
                        sy + 2,
                        fill=source_color,
                        outline="",
                    )
                )
            self._marker_canvas_items.append(
                self.canvas.create_line(
                    tx - target_radius,
                    ty,
                    tx + target_radius,
                    ty,
                    fill=color,
                    width=outline_width,
                )
            )
            self._marker_canvas_items.append(
                self.canvas.create_line(
                    tx,
                    ty - target_radius,
                    tx,
                    ty + target_radius,
                    fill=color,
                    width=outline_width,
                )
            )

    def _update_pan_preview_only(self, state: SliceFitState, shift_xy: tuple[float, float]) -> None:
        self._preview_overlay_shift = (float(shift_xy[0]), float(shift_xy[1]))
        if self._current_atlas_alpha is not None:
            self._apply_atlas_overlay_alpha(self._current_atlas_alpha, shift_xy=shift_xy)
        self._draw_marker_canvas_items(state, source_shift=shift_xy)
        self._update_preview_note()

    def _update_marker_preview_only(self, base_state: SliceFitState, state: SliceFitState) -> None:
        self._preview_overlay_shift = (0.0, 0.0)
        if self._drag_overlay_base_alpha is not None:
            warped_contours = self._warp_boundary_contours_for_marker_preview(self._drag_overlay_base_alpha, base_state, state)
            if warped_contours is not None:
                self._draw_preview_contours(warped_contours, shift_xy=(0.0, 0.0))
            else:
                warped_alpha = self._warp_atlas_alpha_for_marker_preview(self._drag_overlay_base_alpha, base_state, state)
                self._apply_atlas_overlay_alpha(warped_alpha, shift_xy=(0.0, 0.0))
        self._draw_marker_canvas_items(state)
        self._update_preview_note()

    def _remove_marker_near_canvas(self, x: float, y: float, max_distance_px: float) -> bool:
        if not (0 <= self.current_index < len(self.image_paths)):
            return False
        name = self.image_paths[self.current_index].name
        state = self.slice_states.get(name)
        if state is None or not state.markers:
            return False
        target_points = self._marker_target_canvas_points(state)
        if not target_points:
            return False
        distances = [math.hypot(px - float(x), py - float(y)) for px, py in target_points]
        best_index = int(np.argmin(distances))
        if distances[best_index] > max_distance_px:
            return False
        markers = [list(marker) for marker in state.markers]
        markers.pop(best_index)
        origin, u, v = self._vectors_for_state(state)
        self._set_state_for_current_slice(self._copy_state_with_markers(state, markers), schedule=True)
        self._append_log(f"Removed marker on {name}")
        return True

    def _find_marker_near_canvas(self, x: float, y: float, max_distance_px: float) -> int | None:
        if not (0 <= self.current_index < len(self.image_paths)):
            return None
        name = self.image_paths[self.current_index].name
        state = self.slice_states.get(name)
        if state is None or not state.markers:
            return None
        target_points = self._marker_target_canvas_points(state)
        if not target_points:
            return None
        distances = [math.hypot(px - float(x), py - float(y)) for px, py in target_points]
        best_index = int(np.argmin(distances))
        if distances[best_index] > max_distance_px:
            return None
        return best_index

    def _delete_selected_marker(self, _event: object | None = None) -> None:
        if not (0 <= self.current_index < len(self.image_paths)):
            return
        name = self.image_paths[self.current_index].name
        state = self.slice_states.get(name)
        if state is None or not state.markers:
            self.status_var.set("No marker to delete")
            return
        index = self._selected_marker_index
        if index is None and self._last_canvas_xy is not None:
            index = self._find_marker_near_canvas(self._last_canvas_xy[0], self._last_canvas_xy[1], max_distance_px=18.0)
        if index is None or index >= len(state.markers):
            self.status_var.set("No marker selected")
            return
        markers = [list(marker) for marker in state.markers]
        markers.pop(index)
        self._selected_marker_index = None
        self._set_state_for_current_slice(self._copy_state_with_markers(state, markers), schedule=False)
        self._clear_marker_canvas_items()
        self._clear_preview_contour_items()
        self._set_atlas_overlay_visibility(True)
        self._set_preview_mode(False)
        self._schedule_render(interactive=False, delay_ms=1)
        self.status_var.set("Marker deleted")
        self._append_log(f"Deleted marker on {name}")

    def _clear_current_markers(self) -> None:
        if not (0 <= self.current_index < len(self.image_paths)):
            return
        name = self.image_paths[self.current_index].name
        state = self.slice_states[name]
        self._selected_marker_index = None
        self._set_state_for_current_slice(self._copy_state_with_markers(state, []), schedule=False)
        self._clear_marker_canvas_items()
        self._clear_preview_contour_items()
        self._set_atlas_overlay_visibility(True)
        self._set_preview_mode(False)
        self._schedule_render(interactive=False, delay_ms=1)
        self.status_var.set("Cleared markers")
        self._append_log(f"Cleared markers on {name}")

    def _nudge_transform(self, field_label: str, kind: str, direction: int) -> None:
        state = self._read_state_from_controls()
        if state is None:
            return
        delta_sign = 1.0 if direction >= 0 else -1.0
        if field_label == "AP (Bregma um)":
            self._set_preview_mode(False)
            self._adjust_ap(10.0 * delta_sign)
            return
        if field_label in {"Atlas size ML (%)", "Atlas size DV (%)"}:
            ui_delta = 2.0 * delta_sign
            if field_label == "Atlas size ML (%)":
                new_ui = max(5.0, self._span_ml_to_ui_size(state.span_ml) + ui_delta)
                state.span_ml = max(1.0, self._ui_size_to_span_ml(new_ui))
            else:
                new_ui = max(5.0, self._span_dv_to_ui_size(state.span_dv) + ui_delta)
                state.span_dv = max(1.0, self._ui_size_to_span_dv(new_ui))
            state = self._state_from_display_values(
                center_ml=state.center_ml,
                center_ap=state.center_ap,
                center_dv=state.center_dv,
                span_ml=state.span_ml,
                span_dv=state.span_dv,
                tilt_ml_deg=state.tilt_ml_deg,
                tilt_dv_deg=state.tilt_dv_deg,
                roll_deg=state.roll_deg,
                markers=state.markers,
                omit_strokes=state.omit_strokes,
                registration_width=state.registration_width,
                registration_height=state.registration_height,
            )
            self._set_preview_mode(False)
            self._set_state_for_current_slice(state, schedule=True)
            return
        if kind == "linear":
            delta = self._linear_step_vox(20.0) * delta_sign
        else:
            delta = 1.0 * delta_sign
        if field_label == "Center ML":
            state.center_ml += delta
        elif field_label == "Center DV":
            state.center_dv += delta
        elif field_label == "Tilt ML deg":
            state.tilt_ml_deg += delta
        elif field_label == "Tilt DV deg":
            state.tilt_dv_deg += delta
        elif field_label == "Roll deg":
            state = self._rolled_state_from_vectors(state, delta)
            self._set_preview_mode(False)
            self._set_state_for_current_slice(state, schedule=True)
            return
        state = self._state_from_display_values(
            center_ml=state.center_ml,
            center_ap=state.center_ap,
            center_dv=state.center_dv,
            span_ml=state.span_ml,
            span_dv=state.span_dv,
            tilt_ml_deg=state.tilt_ml_deg,
            tilt_dv_deg=state.tilt_dv_deg,
            roll_deg=state.roll_deg,
            markers=state.markers,
            omit_strokes=state.omit_strokes,
            registration_width=state.registration_width,
            registration_height=state.registration_height,
        )
        self._set_preview_mode(False)
        self._set_state_for_current_slice(state, schedule=True)

    def _canvas_render_geometry(self) -> tuple[int, int, float, tuple[int, int]]:
        if self.current_preview_rgb is None:
            return 1, 1, 1.0, (1, 1)
        canvas_w = max(int(self.canvas.winfo_width()), 800)
        canvas_h = max(int(self.canvas.winfo_height()), 600)
        image_h, image_w = self.current_preview_rgb.shape[:2]
        fit_scale = min(
            float(canvas_w) / float(max(image_w, 1)),
            float(canvas_h) / float(max(image_h, 1)),
        )
        scale = max(0.05, fit_scale * self.view_zoom_factor)
        render_shape = (
            max(1, int(round(image_h * scale))),
            max(1, int(round(image_w * scale))),
        )
        return canvas_w, canvas_h, scale, render_shape

    def _prime_display_geometry_for_current_slice(
        self,
        center_view: bool = False,
        display_offset: tuple[float, float] | None = None,
    ) -> None:
        if self.current_preview_rgb is None:
            return
        canvas_w, canvas_h, render_scale, render_shape = self._canvas_render_geometry()
        pad_x = max(24, int(round(render_shape[1] * self.ATLAS_MARGIN_RATIO)))
        pad_y = max(24, int(round(render_shape[0] * self.ATLAS_MARGIN_RATIO)))
        composite_shape = (render_shape[0] + (2 * pad_y), render_shape[1] + (2 * pad_x))
        self.current_preview_scale = render_scale
        self.current_preview_shape = render_shape
        self.current_composite_shape = composite_shape
        self.current_image_offset_in_composite = (float(pad_x), float(pad_y))
        if display_offset is not None:
            self.current_display_offset = (float(display_offset[0]), float(display_offset[1]))
        elif center_view or self.current_display_offset == (0.0, 0.0):
            self.current_display_offset = (
                (canvas_w - composite_shape[1]) * 0.5,
                (canvas_h - composite_shape[0]) * 0.5,
            )

    def _atlas_preview_shape(self, display_shape: tuple[int, int]) -> tuple[int, int]:
        if self._drag_mode in {"marker_move", "omit_draw"}:
            max_edge = self.DISPLAY_PREVIEW_MAX_EDGE_MARKER
        elif self._drag_mode == "pan":
            max_edge = self.DISPLAY_PREVIEW_MAX_EDGE_INTERACTIVE
        else:
            max_edge = self.DISPLAY_PREVIEW_MAX_EDGE
        height, width = display_shape
        longest = max(height, width, 1)
        if longest <= max_edge:
            return display_shape
        scale = float(max_edge) / float(longest)
        return (
            max(1, int(round(height * scale))),
            max(1, int(round(width * scale))),
        )

    def _update_preview_note(self, hover_text: str = "") -> None:
        if not (0 <= self.current_index < len(self.image_paths)):
            self.preview_note_var.set("")
            return
        path = self.image_paths[self.current_index]
        height, width = self.image_sizes[path.name]
        display_name = self.current_display_image_path.name if self.current_display_image_path else path.name
        display_line = f"Display image: {display_name}\n" if display_name != path.name else ""
        note = (
            f"Image {width}x{height} | Zoom {self.view_zoom_factor:.2f}x\n"
            f"{display_line}"
            "Pan: left-drag atlas, mouse wheel zoom\n"
            "Space hold: fade atlas for alignment check | Marker: right-click add, left-drag move, Delete/BackSpace remove\n"
            "Omit: Brush left-drag paint | Polygon left-click add, right-click close | Undo omit removes the last point"
        )
        self.preview_note_var.set(note)

    def _adjust_ap(self, delta: float) -> None:
        state = self._read_state_from_controls()
        if state is None:
            return
        delta_vox = float(delta / max(self._current_target_voxel_size_um(), 1e-6))
        self._set_preview_mode(False)
        self._set_state_for_current_slice(self._translated_state(state, (0.0, delta_vox, 0.0)), schedule=True)
        self.status_var.set("Adjusted AP")

    def _adjust_roll(self, delta: float) -> None:
        state = self._read_state_from_controls()
        if state is None:
            return
        state = self._rolled_state_from_vectors(state, float(delta))
        self._set_preview_mode(False)
        self._set_state_for_current_slice(state, schedule=True)
        self.status_var.set("Adjusted roll")

    def _apply_zoom(self, factor: float) -> None:
        if self.current_preview_rgb is None:
            return
        if self._last_canvas_xy is None:
            anchor_x = float(self.canvas.winfo_width()) * 0.5
            anchor_y = float(self.canvas.winfo_height()) * 0.5
        else:
            anchor_x = self._last_canvas_xy[0]
            anchor_y = self._last_canvas_xy[1]
        image_xy = self._full_image_xy_from_canvas(anchor_x, anchor_y)
        if image_xy is None:
            return
        self.view_zoom_factor = max(0.25, min(6.0, self.view_zoom_factor * factor))
        _canvas_w, _canvas_h, new_scale, new_render_shape = self._canvas_render_geometry()
        new_pad_x = max(24, int(round(new_render_shape[1] * self.ATLAS_MARGIN_RATIO)))
        new_pad_y = max(24, int(round(new_render_shape[0] * self.ATLAS_MARGIN_RATIO)))
        new_display_offset = (
            float(anchor_x) - float(new_pad_x) - (image_xy[0] * new_scale),
            float(anchor_y) - float(new_pad_y) - (image_xy[1] * new_scale),
        )
        self._prime_display_geometry_for_current_slice(center_view=False, display_offset=new_display_offset)
        self._set_preview_mode(False)
        self._marker_input_ready = False
        self.status_var.set(f"Zoom {self.view_zoom_factor:.2f}x")
        self._update_preview_note()
        self._schedule_render()

    def _reset_view_fit(self) -> None:
        if self.current_preview_rgb is None:
            return
        self.view_zoom_factor = 1.0
        self._prime_display_geometry_for_current_slice(center_view=True)
        self._set_preview_mode(False)
        self._marker_input_ready = False
        self.status_var.set("Fit to window")
        self._update_preview_note()
        self._schedule_render()

    def _rotation_matrix(self, state: SliceFitState) -> np.ndarray:
        ax = math.radians(state.tilt_dv_deg)
        ay = math.radians(state.roll_deg)
        az = math.radians(state.tilt_ml_deg)

        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(ax), -math.sin(ax)],
                [0.0, math.sin(ax), math.cos(ax)],
            ],
            dtype=np.float64,
        )
        ry = np.array(
            [
                [math.cos(ay), 0.0, math.sin(ay)],
                [0.0, 1.0, 0.0],
                [-math.sin(ay), 0.0, math.cos(ay)],
            ],
            dtype=np.float64,
        )
        rz = np.array(
            [
                [math.cos(az), -math.sin(az), 0.0],
                [math.sin(az), math.cos(az), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return rz @ rx @ ry

    def _registration_slice_for(
        self,
        path: Path,
        state: SliceFitState,
        use_preview: bool,
        preview_shape: tuple[int, int] | None = None,
        preview_image_shape: tuple[int, int] | None = None,
        preview_padding: tuple[int, int] = (0, 0),
    ) -> RegistrationSlice:
        preview_height, preview_width = preview_shape or self.current_preview_shape or self.image_sizes[path.name]
        original_height, original_width = self.image_sizes[path.name]
        reg_width = max(int(state.registration_width or original_width), 1)
        reg_height = max(int(state.registration_height or original_height), 1)
        out_height = reg_height
        out_width = reg_width
        origin, u, v = self._vectors_for_state(state)
        markers = [list(marker) for marker in state.markers]
        if markers:
            target_scale_x = float(reg_width) / float(max(original_width, 1))
            target_scale_y = float(reg_height) / float(max(original_height, 1))
            normalized_markers: list[list[float]] = []
            for marker in markers:
                if len(marker) < 4:
                    continue
                normalized_markers.append(
                    [
                        float(marker[0]),
                        float(marker[1]),
                        float(marker[2]) * target_scale_x,
                        float(marker[3]) * target_scale_y,
                    ]
                )
            markers = normalized_markers

        return RegistrationSlice(
            filename=path.name,
            nr=self.current_index + 1,
            width=out_width,
            height=out_height,
            origin=origin,
            u=u,
            v=v,
            target_resolution=self.current_target_resolution,
            markers=markers,
            raw={},
        )

    def _schedule_render(self, interactive: bool = False, delay_ms: int | None = None) -> None:
        if not interactive:
            self._marker_input_ready = False
        if self._render_in_progress:
            self._rerender_requested = True
            if not interactive:
                self._rerender_interactive = False
            else:
                self._rerender_interactive = self._rerender_interactive or True
            return
        if self._render_after_id is not None:
            self.after_cancel(self._render_after_id)
            self._render_after_id = None
        self._render_request_id += 1
        request_id = self._render_request_id
        self._render_interactive = bool(interactive)
        if delay_ms is None:
            delay_ms = self.INTERACTIVE_RENDER_DELAY_MS if interactive else self.FINAL_RENDER_DELAY_MS
        self._render_after_id = self.after(int(delay_ms), lambda rid=request_id: self._render_current_slice(rid))

    def _schedule_refine_render(self) -> None:
        if self._refine_after_id is not None:
            self.after_cancel(self._refine_after_id)
        current_request_id = self._render_request_id
        self._refine_after_id = self.after(
            self.INTERACTIVE_REFINE_DELAY_MS,
            lambda rid=current_request_id: self._schedule_render(interactive=False, delay_ms=self.FINAL_RENDER_DELAY_MS)
            if rid == self._render_request_id
            else None,
        )

    def _ensure_canvas_base_layer(
        self,
        path_name: str,
        display_image: np.ndarray,
        render_shape: tuple[int, int],
        composite_shape: tuple[int, int],
        pad_x: int,
        pad_y: int,
    ) -> None:
        signature = (path_name, render_shape, composite_shape)
        if self._current_base_signature == signature and self._canvas_base_item is not None:
            self.canvas.coords(
                self._canvas_base_item,
                self.current_display_offset[0],
                self.current_display_offset[1],
            )
            return
        padded_display_image = np.zeros((composite_shape[0], composite_shape[1], 3), dtype=np.uint8)
        padded_display_image[pad_y : pad_y + render_shape[0], pad_x : pad_x + render_shape[1]] = display_image
        base_image = Image.fromarray(padded_display_image, mode="RGB")
        self.current_base_photo = ImageTk.PhotoImage(base_image)
        if self._canvas_base_item is None:
            self._canvas_base_item = self.canvas.create_image(
                self.current_display_offset[0],
                self.current_display_offset[1],
                anchor="nw",
                image=self.current_base_photo,
            )
        else:
            self.canvas.itemconfigure(self._canvas_base_item, image=self.current_base_photo)
            self.canvas.coords(
                self._canvas_base_item,
                self.current_display_offset[0],
                self.current_display_offset[1],
            )
        self._current_base_signature = signature

    def _render_current_slice(self, request_id: int) -> None:
        self._render_after_id = None
        if request_id != self._render_request_id:
            return
        interactive = bool(self._render_interactive)
        if self.current_preview_rgb is None or not (0 <= self.current_index < len(self.image_paths)):
            return
        self._render_in_progress = True
        state = self._read_state_from_controls()
        try:
            if state is None:
                return
            path = self.image_paths[self.current_index]
            self.slice_states[path.name] = state
            if (
                self.current_preview_shape is None
                or self.current_composite_shape is None
                or self.current_preview_scale <= 0.0
            ):
                self._prime_display_geometry_for_current_slice(center_view=True)
            render_shape = self.current_preview_shape or self.current_preview_rgb.shape[:2]
            composite_shape = self.current_composite_shape or render_shape
            render_scale = float(self.current_preview_scale)
            pad_x = int(round(float(self.current_image_offset_in_composite[0])))
            pad_y = int(round(float(self.current_image_offset_in_composite[1])))
            canvas_w = max(int(self.canvas.winfo_width()), 800)
            canvas_h = max(int(self.canvas.winfo_height()), 600)
            if interactive:
                atlas_preview_shape = self._atlas_preview_shape(composite_shape)
                preview_scale_y = float(atlas_preview_shape[0]) / float(max(composite_shape[0], 1))
                preview_scale_x = float(atlas_preview_shape[1]) / float(max(composite_shape[1], 1))
                atlas_preview_pad_x = max(1, int(round(pad_x * preview_scale_x)))
                atlas_preview_pad_y = max(1, int(round(pad_y * preview_scale_y)))
                preview_render_shape = (
                    max(1, int(round(render_shape[0] * preview_scale_y))),
                    max(1, int(round(render_shape[1] * preview_scale_x))),
                )
            else:
                atlas_preview_shape = composite_shape
                atlas_preview_pad_x = pad_x
                atlas_preview_pad_y = pad_y
                preview_render_shape = render_shape
            preview_slice = self._registration_slice_for(
                path,
                state,
                use_preview=True,
                preview_shape=atlas_preview_shape,
                preview_image_shape=preview_render_shape,
                preview_padding=(atlas_preview_pad_x, atlas_preview_pad_y),
            )
            preview_slice, _, _ = self.atlas.adapted_registration_slice(preview_slice)
            region_map, _, render_metrics = build_registered_maps(
                self.atlas,
                preview_slice,
                atlas_preview_shape,
                midline_threshold_um=75.0,
                registration_offset_px=(float(atlas_preview_pad_x), float(atlas_preview_pad_y)),
                registration_source_shape=preview_render_shape,
                chunk_rows=80 if interactive else 128,
                smooth_regions=not interactive,
                smoothing_kernel_size=5,
                smoothing_iterations=1,
                smoothing_downsample_factor=2,
                simplify_contours=False,
                contour_tolerance_px=2.5,
                contour_min_component_area_px=128,
                atlas_sampling_mode="nearest",
                atlas_sampling_radius_vox=2,
                atlas_sampling_batch_size=4096,
            )
            if atlas_preview_shape != composite_shape:
                region_map = np.asarray(
                    Image.fromarray(region_map.astype(np.int32), mode="I").resize(
                        (composite_shape[1], composite_shape[0]),
                        resample=Image.Resampling.NEAREST,
                    ),
                    dtype=np.int32,
                )
            display_image = self.current_preview_rgb
            if render_shape != self.current_preview_rgb.shape[:2]:
                display_image = np.asarray(
                    Image.fromarray(self.current_preview_rgb).resize(
                        (render_shape[1], render_shape[0]),
                        resample=Image.Resampling.BILINEAR,
                    )
                )
            self.current_region_map = region_map
            self._ensure_canvas_base_layer(
                f"{path.name}|display={self.current_display_image_path.name if self.current_display_image_path else path.name}",
                display_image,
                render_shape,
                composite_shape,
                pad_x,
                pad_y,
            )
            atlas_overlay = self._compose_atlas_overlay(
                region_map,
                self._effective_atlas_opacity(),
                (composite_shape[1], composite_shape[0]),
            )
            self._current_atlas_alpha = np.asarray(atlas_overlay.getchannel("A"), dtype=np.uint8)
            self._cache_boundary_preview_geometry(self._current_atlas_alpha)
            omit_overlay = self._compose_omit_overlay(
                state.omit_strokes,
                (composite_shape[1], composite_shape[0]),
            )
            if (
                self.current_atlas_photo is None
                or self.current_atlas_photo.width() != atlas_overlay.size[0]
                or self.current_atlas_photo.height() != atlas_overlay.size[1]
            ):
                self.current_atlas_photo = ImageTk.PhotoImage(atlas_overlay)
            else:
                self.current_atlas_photo.paste(atlas_overlay)
            if (
                self.current_omit_photo is None
                or self.current_omit_photo.width() != omit_overlay.size[0]
                or self.current_omit_photo.height() != omit_overlay.size[1]
            ):
                self.current_omit_photo = ImageTk.PhotoImage(omit_overlay)
            else:
                self.current_omit_photo.paste(omit_overlay)
            if self._canvas_atlas_item is None:
                self._canvas_atlas_item = self.canvas.create_image(
                    self.current_display_offset[0],
                    self.current_display_offset[1],
                    anchor="nw",
                    image=self.current_atlas_photo,
                )
            else:
                self.canvas.itemconfigure(self._canvas_atlas_item, image=self.current_atlas_photo)
                self.canvas.coords(
                    self._canvas_atlas_item,
                    self.current_display_offset[0],
                    self.current_display_offset[1],
                )
            self._clear_preview_contour_items()
            self._set_atlas_overlay_visibility(True)
            if self._canvas_omit_item is None:
                self._canvas_omit_item = self.canvas.create_image(
                    self.current_display_offset[0],
                    self.current_display_offset[1],
                    anchor="nw",
                    image=self.current_omit_photo,
                )
            else:
                self.canvas.itemconfigure(self._canvas_omit_item, image=self.current_omit_photo)
                self.canvas.coords(
                    self._canvas_omit_item,
                    self.current_display_offset[0],
                    self.current_display_offset[1],
                )
            self._draw_marker_canvas_items(state)
            self.canvas.configure(
                scrollregion=(
                    min(0, self.current_display_offset[0]),
                    min(0, self.current_display_offset[1]),
                    max(canvas_w, self.current_display_offset[0] + composite_shape[1]),
                    max(canvas_h, self.current_display_offset[1] + composite_shape[0]),
                )
            )
            if self._last_canvas_xy is not None:
                self._update_omit_cursor_preview(self._last_canvas_xy[0], self._last_canvas_xy[1])
            self._update_preview_note()
            self._set_preview_mode(False)
            self._marker_input_ready = not interactive
            if state.markers and bool(render_metrics.get("marker_warp_enabled", False)):
                self.status_var.set(f"Rendered marker-warped atlas overlay ({len(state.markers)} markers)")
            else:
                self.status_var.set("Rendered atlas preview" if interactive else "Rendered atlas overlay")
        finally:
            self._render_in_progress = False
            if self._rerender_requested:
                rerender_interactive = self._rerender_interactive
                self._rerender_requested = False
                self._rerender_interactive = False
                self._schedule_render(interactive=rerender_interactive, delay_ms=1)

    def _compose_atlas_overlay(
        self,
        region_map: np.ndarray,
        opacity: float,
        image_size: tuple[int, int],
    ) -> Image.Image:
        return self._compose_boundary_overlay(
            region_map,
            opacity=max(0.20, opacity),
            image_size=image_size,
            color=(95, 210, 255),
            max_filter_size=1 if self._render_interactive else 3,
        )

    def _compose_boundary_overlay(
        self,
        region_map: np.ndarray,
        opacity: float,
        image_size: tuple[int, int],
        color: tuple[int, int, int],
        max_filter_size: int,
    ) -> Image.Image:
        boundaries = self._boundary_mask(region_map)
        boundary_mask = Image.fromarray((boundaries.astype(np.uint8) * 255), mode="L").filter(
            ImageFilter.MaxFilter(max(1, int(max_filter_size)))
        )
        overlay = Image.new("RGBA", image_size, (int(color[0]), int(color[1]), int(color[2]), 0))
        alpha = boundary_mask.point(lambda value: int(min(255, value * max(0.0, opacity))))
        overlay.putalpha(alpha)
        return overlay

    def _compose_omit_overlay(
        self,
        omit_strokes: list[dict[str, object]],
        image_size: tuple[int, int],
    ) -> Image.Image:
        overlay = Image.new("RGBA", image_size, (255, 64, 64, 0))
        if not omit_strokes:
            return overlay
        omit_mask = self._render_omit_mask(
            image_size,
            omit_strokes,
            self.current_preview_scale,
            self.current_image_offset_in_composite,
        )
        omit_alpha = omit_mask.point(lambda value: int(min(255, value * 0.60)))
        overlay.putalpha(omit_alpha)
        return overlay

    def _render_omit_mask(
        self,
        image_size: tuple[int, int],
        omit_strokes: list[dict[str, object]],
        scale: float,
        offset_xy: tuple[float, float] = (0.0, 0.0),
    ) -> Image.Image:
        mask = Image.new("L", image_size, 0)
        draw = PIL.ImageDraw.Draw(mask)
        offset_x = float(offset_xy[0])
        offset_y = float(offset_xy[1])
        for stroke in omit_strokes:
            points = stroke.get("points", [])
            if not isinstance(points, list) or not points:
                continue
            mode = str(stroke.get("mode", "brush")).strip().lower() or "brush"
            width_px = max(1, int(round(float(stroke.get("size", 1.0)) * float(scale))))
            scaled_points = [
                (
                    offset_x + (float(point[0]) * float(scale)),
                    offset_y + (float(point[1]) * float(scale)),
                )
                for point in points
            ]
            if mode == "polygon":
                if len(scaled_points) >= 3:
                    draw.polygon(scaled_points, fill=255)
                elif len(scaled_points) == 2:
                    draw.line(scaled_points, fill=255, width=max(1, width_px // 3), joint="curve")
                else:
                    x, y = scaled_points[0]
                    radius = max(1.0, width_px * 0.25)
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
                continue
            if len(scaled_points) == 1:
                x, y = scaled_points[0]
                radius = max(1.0, width_px * 0.5)
                draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
            else:
                draw.line(scaled_points, fill=255, width=width_px, joint="curve")
                radius = max(1.0, width_px * 0.5)
                for x, y in (scaled_points[0], scaled_points[-1]):
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
        return mask

    def _boundary_mask(self, region_map: np.ndarray) -> np.ndarray:
        positive = region_map > 0
        boundaries = np.zeros_like(region_map, dtype=bool)
        boundaries[1:, :] |= region_map[1:, :] != region_map[:-1, :]
        boundaries[:, 1:] |= region_map[:, 1:] != region_map[:, :-1]
        outer = positive & ~(
            np.pad(positive[1:, :], ((0, 1), (0, 0)), constant_values=False)
            & np.pad(positive[:-1, :], ((1, 0), (0, 0)), constant_values=False)
            & np.pad(positive[:, 1:], ((0, 0), (0, 1)), constant_values=False)
            & np.pad(positive[:, :-1], ((0, 0), (1, 0)), constant_values=False)
        )
        return (boundaries & positive) | outer

    def _step_slice(self, step: int) -> None:
        if not self.image_paths:
            return
        if not self._store_current_slice_state():
            return
        self.current_index = max(0, min(self.current_index + step, len(self.image_paths) - 1))
        self.slice_combo.current(self.current_index)
        self._load_current_slice()

    def _select_slice_from_combo(self, _event: object | None = None) -> None:
        if not self.image_paths:
            return
        if not self._store_current_slice_state():
            if 0 <= self.current_index < len(self.image_paths):
                self.slice_combo.current(self.current_index)
            return
        self.current_index = self.slice_combo.current()
        self._load_current_slice()

    def _copy_to_next(self) -> None:
        if not (0 <= self.current_index < len(self.image_paths) - 1):
            return
        current_name = self.image_paths[self.current_index].name
        next_name = self.image_paths[self.current_index + 1].name
        current_state = self._read_state_from_controls()
        if current_state is None:
            return
        self.slice_states[current_name] = current_state
        self.slice_states[next_name] = self._clone_state(current_state)
        self._append_log(f"Copied transform from {current_name} to {next_name}")

    def _session_payload(self) -> dict[str, object]:
        return {
            "image_folder": self.image_folder_var.get().strip(),
            "input_json": self.input_json_var.get().strip(),
            "output_json": self.output_json_var.get().strip(),
            "atlas_name": self.current_target_name,
            "atlas_target_resolution": [int(value) for value in self.current_target_resolution],
            "atlas_labels_path": str(self.app_config.atlas.labels_path),
            "slice_states": {name: asdict(state) for name, state in self.slice_states.items()},
        }

    def _save_session(self) -> None:
        path_text = self.session_path_var.get().strip()
        if not path_text:
            messagebox.showerror("Missing path", "Session YAML path is required.")
            return
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self._session_payload(), handle, sort_keys=False, allow_unicode=True)
        self._append_log(f"Saved session: {path}")
        self.status_var.set("Session saved")

    def _restore_saved_state(self) -> None:
        if not self.saved_slice_states:
            self.status_var.set("Nothing to restore")
            return
        self.slice_states = {name: self._clone_state(state) for name, state in self.saved_slice_states.items()}
        if self.image_paths:
            self._load_current_slice()
        self.status_var.set("Restored saved state")
        self._append_log("Restored unsaved changes from last saved state")

    def _load_session(self) -> None:
        path_text = self.session_path_var.get().strip()
        if not path_text:
            selected = filedialog.askopenfilename(
                initialdir=self.image_folder_var.get() or os.getcwd(),
                filetypes=[("YAML", "*.yaml *.yml")],
            )
            if not selected:
                return
            self.session_path_var.set(selected)
            path_text = selected
        path = Path(path_text)
        if not path.exists():
            messagebox.showerror("Missing session", f"Session file not found:\n{path}")
            return
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        image_folder = str(payload.get("image_folder", "")).strip()
        input_json = str(payload.get("input_json", "")).strip()
        output_json = str(payload.get("output_json", "")).strip()
        if image_folder:
            self.image_folder_var.set(image_folder)
        if input_json:
            self.input_json_var.set(input_json)
        if output_json:
            self.output_json_var.set(output_json)
        atlas_name = str(payload.get("atlas_name", "")).strip()
        target_resolution = payload.get("atlas_target_resolution")
        if atlas_name:
            self.current_target_name = atlas_name
        if isinstance(target_resolution, (list, tuple)) and len(target_resolution) == 3:
            self.current_target_resolution = tuple(int(value) for value in target_resolution)
        self._load_image_folder(reset_target=not bool(atlas_name or target_resolution))
        state_payload = payload.get("slice_states", {})
        if isinstance(state_payload, dict):
            for name, values in state_payload.items():
                if not isinstance(values, dict) or name not in self.slice_states:
                    continue
                try:
                    self.slice_states[name] = SliceFitState(**values)
                except TypeError:
                    continue
        if self.image_paths:
            self.image_paths = self._sort_paths_anterior_to_posterior(self.image_paths, self.slice_states)
            self.slice_combo["values"] = [path.name for path in self.image_paths]
            self.current_index = 0
            self.slice_combo.current(0)
            self._load_current_slice()
        self._update_saved_snapshot()
        self._append_log(f"Loaded session: {path}")
        self.status_var.set("Session loaded")

    def _load_quicknii_json(self) -> None:
        path_text = self.input_json_var.get().strip()
        if not path_text:
            selected = filedialog.askopenfilename(
                initialdir=self._initial_browse_dir("input_json", self.input_json_var.get(), self.image_folder_var.get()),
                filetypes=[("JSON", "*.json")],
            )
            if not selected:
                return
            self.input_json_var.set(selected)
            self._remember_browse_dir("input_json", selected)
            path_text = selected
        path = Path(path_text)
        if not path.exists():
            messagebox.showerror("Missing JSON", f"JSON file not found:\n{path}")
            return
        self._remember_browse_dir("input_json", path)
        registration = parse_registration_file(path)
        self.current_target_name = registration.target or self.app_config.atlas.name
        self.current_target_resolution = tuple(registration.target_resolution) if registration.target_resolution else tuple(
            int(value) for value in self.app_config.atlas.quicknii_resolution_vox
        )
        self._switch_display_atlas_for_target(self.current_target_name, self.current_target_resolution)
        discovered = self._find_image_folder_for_registration(path, registration)
        wanted_names = {self._normalize_filename_key(item.filename) for item in registration.slices}
        loaded_names = {self._normalize_filename_key(image_path.name) for image_path in self.image_paths}
        current_matches = len(loaded_names & wanted_names)
        should_load_discovered = False
        if discovered is not None:
            try:
                current_folder = Path(self.image_folder_var.get().strip()).resolve()
            except (OSError, RuntimeError):
                current_folder = Path(self.image_folder_var.get().strip())
            try:
                discovered_folder = discovered.resolve()
            except OSError:
                discovered_folder = discovered
            should_load_discovered = (
                not self.image_paths
                or current_matches == 0
                or current_folder != discovered_folder
            )
        if should_load_discovered and discovered is not None:
            self.image_folder_var.set(str(discovered))
            if not self.output_json_var.get().strip() or current_matches == 0:
                self.output_json_var.set(str(path))
            if not self.session_path_var.get().strip():
                self.session_path_var.set(str(path.with_suffix(".session.yaml")))
            self._load_image_folder(reset_target=False)
        elif not self.image_paths:
            messagebox.showerror(
                "Missing images",
                "Could not auto-detect an image folder for this JSON.\n"
                "Place the JSON next to its images or select a matching JSON.",
            )
            return
        output_candidate_text = self.output_json_var.get().strip()
        omit_by_name: dict[str, list[dict[str, object]]] = {}
        if output_candidate_text:
            omit_by_name = self._load_omit_sidecar(Path(output_candidate_text))
        if not omit_by_name:
            omit_by_name = self._load_omit_sidecar(path)
        loaded = 0
        image_by_name = {self._normalize_filename_key(image_path.name): image_path for image_path in self.image_paths}
        matched_paths: list[Path] = []
        matched_sizes: dict[str, tuple[int, int]] = {}
        matched_states: dict[str, SliceFitState] = {}
        missing_images: list[str] = []
        for registration_slice in registration.slices:
            image_path = image_by_name.get(self._normalize_filename_key(registration_slice.filename))
            if image_path is None:
                missing_images.append(registration_slice.filename)
                continue
            matched_paths.append(image_path)
            matched_sizes[image_path.name] = self.image_sizes[image_path.name]
            matched_states[image_path.name] = self._state_from_registration_slice(
                registration_slice,
                omit_strokes=self._omit_strokes_for_image(omit_by_name, image_path.name),
            )
            loaded += 1
        if loaded == 0:
            messagebox.showwarning("No matches", "No slice filenames in the JSON matched the loaded image folder.")
            return
        matched_paths = self._sort_paths_anterior_to_posterior(matched_paths, matched_states)
        self.image_paths = matched_paths
        self.image_sizes = matched_sizes
        self.slice_states = matched_states
        self.slice_combo["values"] = [path.name for path in self.image_paths]
        self.current_index = 0
        self.slice_combo.current(0)
        self._load_current_slice()
        self._update_saved_snapshot()
        try:
            self._last_loaded_input_json = str(path.resolve())
        except OSError:
            self._last_loaded_input_json = str(path)
        self._append_log(f"Loaded {loaded} slice fits from {path}")
        marker_count = sum(len(state.markers) for state in matched_states.values())
        if marker_count:
            marked_slices = sum(1 for state in matched_states.values() if state.markers)
            self._append_log(f"Loaded {marker_count} AtlasFitter markers on {marked_slices} slices")
        if missing_images:
            self._append_log(f"Skipped {len(missing_images)} JSON slices with no matching image files")
        self.status_var.set("JSON loaded")

    def _find_image_folder_for_registration(
        self,
        json_path: Path,
        registration,
    ) -> Path | None:
        wanted = {self._normalize_filename_key(item.filename) for item in registration.slices}
        if not wanted:
            return None

        candidates: list[Path] = []
        for candidate in (
            json_path.parent,
            json_path.parent / "raw",
            json_path.parent / "jpg",
            json_path.parent / "png",
            json_path.parent.parent,
            json_path.parent.parent / "raw",
            json_path.parent.parent / "jpg",
            json_path.parent.parent / "png",
        ):
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            if resolved.exists() and resolved.is_dir() and resolved not in candidates:
                candidates.append(resolved)

        best_dir: Path | None = None
        best_score = -1
        for candidate in candidates:
            names = {self._normalize_filename_key(path.name) for path in self._list_image_paths(candidate)}
            score = len(names & wanted)
            if score > best_score:
                best_score = score
                best_dir = candidate
        return best_dir if best_score > 0 else None

    def _save_omit_sidecar(self, json_path: Path) -> None:
        masks_dir = self._omit_masks_dir(json_path)
        state_path = self._omit_state_path(json_path)
        sidecar_slices: dict[str, dict[str, object]] = {}
        keep_files: set[str] = set()
        masks_dir.mkdir(parents=True, exist_ok=True)
        for image_path in self.image_paths:
            state = self.slice_states.get(image_path.name)
            if state is None or not state.omit_strokes:
                continue
            image_height, image_width = self.image_sizes[image_path.name]
            mask = self._render_omit_mask((image_width, image_height), state.omit_strokes, 1.0)
            slice_key = self._channel_neutral_filename_key(image_path.name)
            mask_stem = slice_key or self._normalize_filename_key(image_path.name)
            mask_name = f"{mask_stem}_omit_mask.png"
            mask_path = masks_dir / mask_name
            mask.save(mask_path)
            keep_files.add(mask_name)
            sidecar_slices[image_path.name] = {
                "slice_key": slice_key,
                "image_filename": image_path.name,
                "omit_strokes": [self._clone_omit_stroke(stroke) for stroke in state.omit_strokes],
                "mask_file": mask_name,
            }
        for stale_mask in masks_dir.glob("*_omit_mask.png"):
            if stale_mask.name not in keep_files:
                stale_mask.unlink(missing_ok=True)
        if sidecar_slices:
            payload = {
                "version": APP_VERSION,
                "source_json": self.input_json_var.get().strip(),
                "output_json": str(json_path),
                "slices": sidecar_slices,
            }
            self._atomic_write_json(state_path, payload)
        else:
            state_path.unlink(missing_ok=True)
        if masks_dir.exists() and not any(masks_dir.iterdir()):
            masks_dir.rmdir()

    def _load_omit_sidecar(self, json_path: Path) -> dict[str, list[dict[str, object]]]:
        state_path = self._omit_state_path(json_path)
        if not state_path.exists():
            return {}
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        slices_payload = payload.get("slices", {})
        if not isinstance(slices_payload, dict):
            return {}
        loaded: dict[str, list[dict[str, object]]] = {}
        for filename, values in slices_payload.items():
            if not isinstance(values, dict):
                continue
            omit_strokes = values.get("omit_strokes", [])
            if not isinstance(omit_strokes, list):
                continue
            cloned_strokes = [self._clone_omit_stroke(stroke) for stroke in omit_strokes if isinstance(stroke, dict)]
            if not cloned_strokes:
                continue
            raw_filename = str(filename)
            keys = {
                raw_filename,
                self._normalize_filename_key(raw_filename),
                self._channel_neutral_filename_key(raw_filename),
            }
            sidecar_slice_key = values.get("slice_key", "")
            if isinstance(sidecar_slice_key, str) and sidecar_slice_key.strip():
                keys.add(sidecar_slice_key.strip().lower().replace(" ", "_"))
            image_filename = values.get("image_filename", "")
            if isinstance(image_filename, str) and image_filename.strip():
                keys.add(image_filename)
                keys.add(self._normalize_filename_key(image_filename))
                keys.add(self._channel_neutral_filename_key(image_filename))
            for key in keys:
                if key and key not in loaded:
                    loaded[key] = [self._clone_omit_stroke(stroke) for stroke in cloned_strokes]
        return loaded

    def _save_quicknii_json(self, silent: bool = False) -> bool:
        if not self.image_paths:
            if not silent:
                messagebox.showerror("No images", "Load an image folder first.")
            return False
        path_text = self.output_json_var.get().strip()
        if not path_text:
            if not silent:
                messagebox.showerror("Missing output", "Output JSON path is required.")
            return False
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        if 0 <= self.current_index < len(self.image_paths):
            state = self._read_state_from_controls()
            if state is not None:
                self.slice_states[self.image_paths[self.current_index].name] = state
        slices = []
        for index, image_path in enumerate(self.image_paths, start=1):
            state = self.slice_states[image_path.name]
            reg_slice = self._registration_slice_for(image_path, state, use_preview=False)
            payload_slice = (
                {
                    "filename": image_path.name,
                    "nr": index,
                    "width": reg_slice.width,
                    "height": reg_slice.height,
                    "anchoring": [float(value) for value in reg_slice.origin] + [float(value) for value in reg_slice.u] + [float(value) for value in reg_slice.v],
                }
            )
            if state.markers:
                payload_slice["markers"] = [[float(value) for value in marker] for marker in state.markers]
            slices.append(payload_slice)
        payload = {
            "name": path.stem,
            "target": self.current_target_name,
            "target-resolution": [int(value) for value in self.current_target_resolution],
            "slices": slices,
        }
        self._atomic_write_json(path, payload)
        self._remember_browse_dir("output_json", path)
        self._save_omit_sidecar(path)
        self._update_saved_snapshot()
        if not silent:
            self._append_log(f"Saved JSON: {path}")
            self.status_var.set("JSON saved")
        return True

    def _store_current_slice_state(self) -> bool:
        if not self.image_paths or not (0 <= self.current_index < len(self.image_paths)):
            return True
        state = self._read_state_from_controls()
        if state is None:
            messagebox.showerror("Invalid transform", "Could not read the current transform. Slice change was cancelled.")
            return False
        self.slice_states[self.image_paths[self.current_index].name] = state
        return True

    def _state_from_registration_slice(
        self,
        registration_slice: RegistrationSlice,
        omit_strokes: list[dict[str, object]] | None = None,
    ) -> SliceFitState:
        return self._state_from_vectors(
            np.asarray(registration_slice.origin, dtype=np.float64),
            np.asarray(registration_slice.u, dtype=np.float64),
            np.asarray(registration_slice.v, dtype=np.float64),
            markers=[list(marker) for marker in registration_slice.markers],
            omit_strokes=[self._clone_omit_stroke(stroke) for stroke in (omit_strokes or [])],
            registration_width=registration_slice.width,
            registration_height=registration_slice.height,
        )


def launch_gui(app_name: str = "QDFevo_2_AtlasFitter") -> None:
    app = AtlasFittingApp(app_name=app_name)
    app.mainloop()
