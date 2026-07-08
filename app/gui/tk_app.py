"""Tkinter desktop application for QUINTdeepflow quantification."""

from __future__ import annotations

import os
import queue
import re
import threading
import tkinter as tk
import itertools
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import numpy as np

from app_version import APP_VERSION, version_label
from config.settings import AppConfig, load_app_config, save_app_config, save_gui_visible_app_config
from io_utils.discovery import discovery_to_dataframe, discover_section_groups, extract_image_channel
from io_utils.portable_bundle import export_portable_bundle
from quantification.pipeline import QuantificationPipeline
from registration.parser import parse_registration_file

OVERLAP_CHANNEL_PATTERN = re.compile(r"CH\d+", re.IGNORECASE)


class QuintDeepflowApp(tk.Tk):
    """Windows-friendly GUI focused on QUINTdeepflow quantification runs."""

    PLACEHOLDER_FG = "#8c8c8c"
    ENTRY_FG = "#111111"
    WINDOW_GEOMETRY = "1420x900"
    WINDOW_MINSIZE = (1200, 760)

    def __init__(self) -> None:
        super().__init__()
        self.withdraw()
        self.export_patch_ids = str(os.environ.get("QUINTDEEPFLOW_EXPORT_PATCH_IDS", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.app_title = str(os.environ.get("QUINTDEEPFLOW_GUI_TITLE", "QUINTdeepflow2")).strip() or "QUINTdeepflow2"
        self.input_json_label = str(os.environ.get("QUINTDEEPFLOW_INPUT_JSON_LABEL", "3. Atlas JSON")).strip() or "3. Atlas JSON"
        self.input_json_check_label = (
            str(os.environ.get("QUINTDEEPFLOW_INPUT_JSON_CHECK_LABEL", "Atlas JSON Check")).strip()
            or "Atlas JSON Check"
        )
        self.input_json_short_label = "Input JSON" if "Input JSON" in self.input_json_label else "Atlas JSON"
        self.title(version_label(self.app_title))
        self.geometry(self.WINDOW_GEOMETRY)
        self.minsize(*self.WINDOW_MINSIZE)
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.default_config_path = Path("sample_configs/default_config.yaml").resolve()
        self.current_config_path = self.default_config_path
        self.result_output_dir: Path | None = None
        self._worker_thread: threading.Thread | None = None
        self._run_in_progress = False
        self._active_output_dir: Path | None = None
        self._configure_style()
        self._build_variables()
        self._build_layout()
        self._load_initial_config()
        self._stabilize_window()
        self.protocol("WM_DELETE_WINDOW", self._on_close_request)
        self.after(150, self._poll_queue)
        self.deiconify()

    def _configure_style(self) -> None:
        self.option_add("*Font", "{Segoe UI} 10")
        self.option_add("*TCombobox*Listbox.font", "{Segoe UI} 10")
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("TButton", padding=(10, 4))
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=24)

    def _stabilize_window(self) -> None:
        self.update_idletasks()
        self.geometry(self.WINDOW_GEOMETRY)
        self.minsize(*self.WINDOW_MINSIZE)

    def _build_variables(self) -> None:
        self.image_folder_var = tk.StringVar()
        self.segmentation_folder_var = tk.StringVar()
        self.atlas_json_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()
        self.sample_name_var = tk.StringVar(value="sample_run")
        self.atlas_default_var = tk.StringVar(value="QUINTdeepflow atlas: Allen mouse default")
        self.status_var = tk.StringVar(value="Ready")
        self.border_assignment_var = tk.StringVar(value="bigger")
        self.parallel_workers_var = tk.StringVar(value="0")
        self.normalize_ilastik_masks_var = tk.BooleanVar(value=False)

        self.channel_rows: list[dict[str, tk.StringVar]] = []
        for _ in range(4):
            self.channel_rows.append(
                {
                    "channel": tk.StringVar(),
                    "min_area": tk.StringVar(),
                    "max_area": tk.StringVar(),
                    "watershed": tk.BooleanVar(value=False),
                    "watershed_marker_threshold_px": tk.StringVar(value="1.5"),
                    "watershed_selective_area_percentile": tk.StringVar(value="90"),
                    "watershed_selective_elongation_threshold": tk.StringVar(value="2.0"),
                }
            )

        self.overlap_rule_rows: list[dict[str, object]] = []
        self.overlap_set_rows: list[dict[str, object]] = []

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        config_frame = ttk.LabelFrame(self, text="QUINTdeepflow Quantification + Visualise")
        config_frame.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        config_frame.columnconfigure(0, minsize=250)
        config_frame.columnconfigure(1, weight=1)
        config_frame.columnconfigure(2, minsize=94)

        self._path_row(config_frame, 0, "1. Input raw image folder", self.image_folder_var, self._browse_image_folder)
        self._path_row(
            config_frame,
            1,
            "2. ilastik segmentation folder",
            self.segmentation_folder_var,
            self._browse_segmentation_folder,
        )
        self._file_row(config_frame, 2, self.input_json_label, self.atlas_json_var, self._browse_atlas_json)
        self._path_row(config_frame, 3, "4. Output folder", self.output_folder_var, self._browse_output_folder)
        self._simple_row(config_frame, 4, "5. Output sample name", self.sample_name_var)

        for index, row_vars in enumerate(self.channel_rows, start=1):
            self._channel_row(config_frame, 4 + index, f"{5 + index}. Channel setting {index}", row_vars)

        ttk.Label(config_frame, text="10. Overlap rules").grid(row=10, column=0, sticky="nw", padx=6, pady=4)
        rule_block = ttk.Frame(config_frame)
        rule_block.grid(row=10, column=1, columnspan=2, sticky="ew", pady=4)
        rule_block.columnconfigure(0, weight=1)
        self.overlap_rule_rows_container = ttk.Frame(rule_block)
        self.overlap_rule_rows_container.grid(row=0, column=0, sticky="ew")
        rule_actions = ttk.Frame(rule_block)
        rule_actions.grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Button(rule_actions, text="Add rule", command=self._add_overlap_rule_row, width=12).pack(side="left", padx=(0, 6))
        ttk.Button(rule_actions, text="Remove last", command=self._remove_overlap_rule_row, width=12).pack(side="left")
        self._ensure_overlap_rule_row_count(2)

        ttk.Label(config_frame, text="11. Overlap set").grid(row=11, column=0, sticky="nw", padx=6, pady=4)
        set_block = ttk.Frame(config_frame)
        set_block.grid(row=11, column=1, columnspan=2, sticky="ew", pady=4)
        set_block.columnconfigure(0, weight=1)
        self.overlap_set_rows_container = ttk.Frame(set_block)
        self.overlap_set_rows_container.grid(row=0, column=0, sticky="ew")
        self._ensure_overlap_set_row_count(1)

        self._options_row(config_frame, 12)

        atlas_note = ttk.Label(
            config_frame,
            textvariable=self.atlas_default_var,
            foreground="#375a7f",
            wraplength=1260,
            justify="left",
        )
        atlas_note.grid(row=13, column=0, columnspan=4, sticky="w", padx=6, pady=(6, 0))

        button_frame = ttk.Frame(config_frame)
        button_frame.grid(row=14, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        for text, command in (
            ("Load Config", self._load_config_dialog),
            ("Save Config", self._save_config_dialog),
            (self.input_json_check_label, self._discover),
            ("Run", self._run_pipeline),
            ("Open Results", self._open_results),
            ("Export Portable", self._export_portable_bundle),
        ):
            ttk.Button(button_frame, text=text, command=command, width=16).pack(side="left", padx=4)

        body = ttk.Frame(self)
        body.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=3, minsize=360)
        body.rowconfigure(1, weight=1)

        table_frame = ttk.LabelFrame(body, text=f"{self.input_json_check_label} / Resolved Bundles")
        table_frame.grid(row=0, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = ("animal_id", "section_id", "channel", "image_file", "mask_file", "registration_json", "registration_entry")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        column_widths = {
            "animal_id": 100,
            "section_id": 100,
            "channel": 90,
            "image_file": 320,
            "mask_file": 320,
            "registration_json": 280,
            "registration_entry": 210,
        }
        for column in columns:
            self.tree.heading(column, text=column)
            self.tree.column(
                column,
                width=column_widths[column],
                minwidth=column_widths[column],
                stretch=False,
            )
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        tree_scroll_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        tree_scroll_x.grid(row=1, column=0, sticky="ew")
        self.tree.configure(yscrollcommand=tree_scroll.set, xscrollcommand=tree_scroll_x.set)

        log_frame = ttk.LabelFrame(body, text="Log")
        log_frame.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = ScrolledText(log_frame, wrap="word", height=7, font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky="nsew")

        status_frame = ttk.Frame(self)
        status_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        status_frame.columnconfigure(1, weight=1)
        status_frame.columnconfigure(2, weight=0)
        ttk.Label(status_frame, textvariable=self.status_var, width=52, anchor="w").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 8),
        )
        self.progress = ttk.Progressbar(status_frame, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=1, sticky="ew")
        ttk.Label(status_frame, text=f"Version {APP_VERSION}", anchor="e", foreground="#5f6b7a").grid(
            row=0,
            column=2,
            sticky="e",
            padx=(8, 0),
        )

    def _path_row(self, parent: ttk.LabelFrame, row: int, label: str, variable: tk.StringVar, browse_command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=(0, 6), pady=4)
        ttk.Button(parent, text="Browse", command=browse_command).grid(row=row, column=2, sticky="ew", padx=(0, 6), pady=4)

    def _file_row(self, parent: ttk.LabelFrame, row: int, label: str, variable: tk.StringVar, browse_command) -> None:
        self._path_row(parent, row, label, variable, browse_command)

    def _simple_row(self, parent: ttk.LabelFrame, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=(0, 6), pady=4)

    def _channel_row(self, parent: ttk.LabelFrame, row: int, label: str, row_vars: dict[str, tk.StringVar]) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Label(frame, text="Use CH").pack(side="left", padx=(0, 4))
        ttk.Combobox(
            frame,
            textvariable=row_vars["channel"],
            values=["", "CH1", "CH2", "CH3", "CH4"],
            width=8,
            state="readonly",
        ).pack(side="left", padx=(0, 12))
        ttk.Label(frame, text="Min area").pack(side="left", padx=(0, 4))
        ttk.Entry(frame, textvariable=row_vars["min_area"], width=10).pack(side="left", padx=(0, 12))
        ttk.Label(frame, text="Max area").pack(side="left", padx=(0, 4))
        ttk.Entry(frame, textvariable=row_vars["max_area"], width=10).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(frame, text="Watershed", variable=row_vars["watershed"]).pack(side="left", padx=(0, 8))
        ttk.Label(frame, text="WS marker th px / auto").pack(side="left", padx=(0, 4))
        ttk.Entry(frame, textvariable=row_vars["watershed_marker_threshold_px"], width=8).pack(side="left", padx=(0, 12))
        ttk.Label(frame, text="WS area pct").pack(side="left", padx=(0, 4))
        ttk.Entry(frame, textvariable=row_vars["watershed_selective_area_percentile"], width=6).pack(side="left", padx=(0, 12))
        ttk.Label(frame, text="WS elong >=").pack(side="left", padx=(0, 4))
        ttk.Entry(frame, textvariable=row_vars["watershed_selective_elongation_threshold"], width=6).pack(side="left")

    def _pair_values(self) -> list[str]:
        channels = [f"CH{index}" for index in range(1, 5)]
        return ["", *[f"{pair[0]}, {pair[1]}" for pair in itertools.combinations(channels, 2)]]

    def _new_overlap_rule_row_vars(self) -> dict[str, object]:
        return {
            "pair": tk.StringVar(),
            "min_overlap_px": tk.StringVar(value="0"),
            "center_distance_px": tk.StringVar(value="0"),
        }

    def _new_overlap_set_row_vars(self) -> dict[str, object]:
        return {f"CH{index}": tk.StringVar(value="") for index in range(1, 5)}

    def _overlap_sign_values(self) -> list[str]:
        return ["", "+", "-", "+-"]

    def _normalize_overlap_selector(self, value: str) -> str:
        stripped = str(value).strip()
        return stripped if stripped in {"+", "-", "+-"} else ""

    def _ensure_overlap_rule_row_count(self, count: int) -> None:
        while len(self.overlap_rule_rows) < count:
            self.overlap_rule_rows.append(self._new_overlap_rule_row_vars())
        while len(self.overlap_rule_rows) > count:
            self.overlap_rule_rows.pop()
        self._render_overlap_rule_rows()

    def _ensure_overlap_set_row_count(self, count: int) -> None:
        while len(self.overlap_set_rows) < count:
            self.overlap_set_rows.append(self._new_overlap_set_row_vars())
        while len(self.overlap_set_rows) > count:
            self.overlap_set_rows.pop()
        self._render_overlap_set_rows()

    def _add_overlap_rule_row(self) -> None:
        self.overlap_rule_rows.append(self._new_overlap_rule_row_vars())
        self._render_overlap_rule_rows()

    def _remove_overlap_rule_row(self) -> None:
        if len(self.overlap_rule_rows) <= 1:
            return
        self.overlap_rule_rows.pop()
        self._render_overlap_rule_rows()

    def _add_overlap_set_row(self) -> None:
        self._ensure_overlap_set_row_count(1)

    def _remove_overlap_set_row(self) -> None:
        self._ensure_overlap_set_row_count(1)

    def _render_overlap_rule_rows(self) -> None:
        for child in self.overlap_rule_rows_container.winfo_children():
            child.destroy()
        for index, row_vars in enumerate(self.overlap_rule_rows, start=1):
            self._overlap_rule_row(self.overlap_rule_rows_container, index - 1, f"Rule {index}", row_vars)

    def _render_overlap_set_rows(self) -> None:
        for child in self.overlap_set_rows_container.winfo_children():
            child.destroy()
        for index, row_vars in enumerate(self.overlap_set_rows, start=1):
            self._overlap_set_row(self.overlap_set_rows_container, index - 1, f"Set {index}", row_vars)

    def _overlap_rule_row(self, parent, row: int, label: str, row_vars: dict[str, object]) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Label(frame, text="Pair").pack(side="left", padx=(0, 4))
        ttk.Combobox(
            frame,
            textvariable=row_vars["pair"],
            values=self._pair_values(),
            width=12,
            state="readonly",
        ).pack(side="left", padx=(0, 12))
        ttk.Label(frame, text="Min overlap pix").pack(side="left", padx=(0, 4))
        ttk.Entry(frame, textvariable=row_vars["min_overlap_px"], width=8).pack(side="left", padx=(0, 12))
        ttk.Label(frame, text="Max center dist pix (if not overlap)").pack(side="left", padx=(0, 4))
        ttk.Entry(frame, textvariable=row_vars["center_distance_px"], width=8).pack(side="left")

    def _overlap_set_row(self, parent, row: int, label: str, row_vars: dict[str, object]) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        for index in range(1, 5):
            channel = f"CH{index}"
            ttk.Label(frame, text=channel).pack(side="left", padx=(0, 4))
            ttk.Combobox(
                frame,
                textvariable=row_vars[channel],
                values=self._overlap_sign_values(),
                width=4,
                state="readonly",
            ).pack(side="left", padx=(0, 10))

    def _options_row(self, parent: ttk.LabelFrame, row: int) -> None:
        ttk.Label(parent, text="12. Border / mask options").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Label(frame, text="Border cell rule").pack(side="left", padx=(0, 4))
        ttk.Combobox(
            frame,
            textvariable=self.border_assignment_var,
            values=["bigger", "center", "omit"],
            width=10,
            state="readonly",
        ).pack(side="left", padx=(0, 12))
        ttk.Label(frame, text="Parallel workers").pack(side="left", padx=(0, 4))
        ttk.Entry(frame, textvariable=self.parallel_workers_var, width=8).pack(side="left", padx=(0, 4))
        ttk.Label(frame, text="0 = auto").pack(side="left", padx=(0, 12))
        ttk.Checkbutton(
            frame,
            text="Auto-normalize ilastik masks to 0/255 (cells white, overwrite)",
            variable=self.normalize_ilastik_masks_var,
        ).pack(side="left")

    def _apply_placeholder(self, entry: tk.Entry, variable: tk.StringVar, placeholder: str) -> None:
        def on_focus_in(_event=None) -> None:
            if entry.cget("fg") == self.PLACEHOLDER_FG and variable.get() == placeholder:
                variable.set("")
                entry.config(fg=self.ENTRY_FG)

        def on_focus_out(_event=None) -> None:
            if not variable.get().strip():
                variable.set(placeholder)
                entry.config(fg=self.PLACEHOLDER_FG)
            else:
                entry.config(fg=self.ENTRY_FG)

        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)
        on_focus_out()

    def _value_without_placeholder(self, value: str, placeholder: str) -> str:
        stripped = str(value).strip()
        return "" if stripped == placeholder else stripped

    def _load_initial_config(self) -> None:
        if self.default_config_path.exists():
            self._apply_config(load_app_config(self.default_config_path))
        self._apply_environment_overrides()
        if not self.output_folder_var.get().strip():
            self.output_folder_var.set(str((Path.cwd() / "outputs").resolve()))

    def _apply_environment_overrides(self) -> None:
        text_overrides = [
            (self.image_folder_var, ["QUINTDEEPFLOW_IMAGE_FOLDER", "QUINTNEXT_IMAGE_FOLDER"]),
            (self.segmentation_folder_var, ["QUINTDEEPFLOW_SEGMENTATION_FOLDER", "QUINTNEXT_SEGMENTATION_FOLDER"]),
            (self.atlas_json_var, ["QUINTDEEPFLOW_ATLAS_JSON", "QUINTNEXT_ATLAS_JSON"]),
            (self.output_folder_var, ["QUINTDEEPFLOW_OUTPUT_FOLDER", "QUINTNEXT_OUTPUT_FOLDER"]),
            (self.sample_name_var, ["QUINTDEEPFLOW_SAMPLE_NAME", "QUINTNEXT_SAMPLE_NAME"]),
            (self.parallel_workers_var, ["QUINTDEEPFLOW_PARALLEL_WORKERS", "QUINTNEXT_PARALLEL_WORKERS"]),
        ]
        for variable, env_names in text_overrides:
            for env_name in env_names:
                value = os.environ.get(env_name, "").strip()
                if value:
                    variable.set(value)
                    break

        project_root = os.environ.get("QUINTDEEPFLOW_PROJECT_ROOT", "").strip()
        if not project_root:
            project_root = os.environ.get("QUINTNEXT_PROJECT_ROOT", "").strip()
        if project_root and not self.output_folder_var.get().strip():
            self.output_folder_var.set(project_root)

        analysis_value = os.environ.get("QUINTDEEPFLOW_ANALYSIS_CHANNELS", "").strip()
        if not analysis_value:
            analysis_value = os.environ.get("QUINTNEXT_ANALYSIS_CHANNELS", "").strip()
        analysis_channels = [item.strip().upper() for item in analysis_value.split(",") if item.strip()]
        if analysis_channels:
            self._set_channel_rows(analysis_channels, {}, {}, {}, {}, {}, {}, False, 1.5, 90.0, 2.0)

        min_area_value = os.environ.get("QUINTDEEPFLOW_MIN_AREA_BY_CH", "").strip()
        if not min_area_value:
            min_area_value = os.environ.get("QUINTNEXT_MIN_AREA_BY_CH", "").strip()
        min_area_mapping = self._parse_numeric_mapping(min_area_value)
        if min_area_mapping:
            current_channels = [row["channel"].get().strip().upper() for row in self.channel_rows if row["channel"].get().strip()]
            ordered = current_channels or list(min_area_mapping)
            self._set_channel_rows(ordered, min_area_mapping, {}, {}, {}, {}, {}, False, 1.5, 90.0, 2.0)

        pair_rules_value = os.environ.get("QUINTDEEPFLOW_PAIR_RULES", "").strip()
        if not pair_rules_value:
            pair_rules_value = os.environ.get("QUINTNEXT_PAIR_RULES", "").strip()
        pair_rules = self._parse_pair_rules(pair_rules_value)
        overlap_value = os.environ.get("QUINTDEEPFLOW_OVERLAP_SETS", "").strip()
        if not overlap_value:
            overlap_value = os.environ.get("QUINTNEXT_OVERLAP_SETS", "").strip()
        overlap_sets = [item.strip() for item in overlap_value.split(",") if item.strip()]
        if pair_rules:
            self._set_overlap_rule_rows(pair_rules)
        if overlap_sets:
            self._set_overlap_set_rows(overlap_sets)

    def _browse_image_folder(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.image_folder_var.get() or os.getcwd())
        if selected:
            self.image_folder_var.set(selected)

    def _browse_segmentation_folder(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.segmentation_folder_var.get() or os.getcwd())
        if selected:
            self.segmentation_folder_var.set(selected)

    def _browse_atlas_json(self) -> None:
        selected = filedialog.askopenfilename(
            initialdir=self.atlas_json_var.get() or self.image_folder_var.get() or os.getcwd(),
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if selected:
            self.atlas_json_var.set(selected)

    def _browse_output_folder(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.output_folder_var.get() or os.getcwd())
        if selected:
            self.output_folder_var.set(selected)

    def _split_csv(self, value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item.strip()]

    def _safe_int(self, value: str, default: int = 0) -> int:
        stripped = str(value).strip()
        if not stripped:
            return default
        return int(float(stripped))

    def _safe_float(self, value: str, default: float = 0.0) -> float:
        stripped = str(value).strip()
        if not stripped:
            return default
        return float(stripped)

    def _parse_numeric_mapping(self, value: str) -> dict[str, int]:
        mapping: dict[str, int] = {}
        for item in self._split_csv(value):
            if ":" not in item:
                continue
            key, raw_value = item.split(":", 1)
            key = key.strip().upper()
            if not key:
                continue
            mapping[key] = self._safe_int(raw_value, 0)
        return mapping

    def _parse_pair_rules(self, value: str) -> dict[str, dict[str, float | int | str]]:
        rules: dict[str, dict[str, float | int | str]] = {}
        for chunk in [item.strip() for item in value.split(";") if item.strip()]:
            if "=" not in chunk:
                continue
            combo, payload = chunk.split("=", 1)
            channels = sorted({channel.upper() for channel in OVERLAP_CHANNEL_PATTERN.findall(combo)}, key=str.upper)
            if len(channels) != 2:
                continue
            combo = "+".join(channels)
            fields = [item.strip() for item in payload.split(":")]
            method = fields[0] if fields and fields[0] else "centroid_distance"
            rule: dict[str, float | int | str] = {"method": method}
            if len(fields) >= 2 and fields[1]:
                threshold = float(fields[1])
                if method == "mask_overlap":
                    rule["overlap_threshold_px"] = int(round(threshold))
                else:
                    rule["distance_threshold_px"] = threshold
            if len(fields) >= 3 and fields[2]:
                rule["fallback_distance_threshold_px"] = float(fields[2])
            rules[combo] = rule
        return rules

    def _pair_display_to_key(self, value: str) -> str:
        channels = sorted({channel.upper() for channel in OVERLAP_CHANNEL_PATTERN.findall(value)}, key=str.upper)
        return "+".join(channels) if len(channels) == 2 else ""

    def _overlap_set_row_to_spec(self, row: dict[str, object]) -> str:
        tokens: list[str] = []
        for index in range(1, 5):
            channel = f"CH{index}"
            selection = self._normalize_overlap_selector(row[channel].get())
            if not selection:
                continue
            tokens.append(f"{channel}{selection}")
        return "".join(tokens)

    def _apply_overlap_set_spec_to_row(self, row: dict[str, object], spec: str) -> None:
        for index in range(1, 5):
            row[f"CH{index}"].set("")
        for channel, sign in re.findall(r"(CH\d+)(\+\-|\+|-|±)", str(spec).upper()):
            normalized = "+-" if sign == "±" else sign
            if channel in row:
                row[channel].set(normalized)

    def _base_config(self) -> AppConfig:
        if self.current_config_path.exists():
            return load_app_config(self.current_config_path)
        if self.default_config_path.exists():
            return load_app_config(self.default_config_path)
        return AppConfig()

    def _selected_channel_specs(
        self,
    ) -> tuple[
        list[str],
        dict[str, int],
        dict[str, int],
        dict[str, bool],
        dict[str, float | str],
        dict[str, float],
        dict[str, float],
    ]:
        ordered_channels: list[str] = []
        min_area_map: dict[str, int] = {}
        max_area_map: dict[str, int] = {}
        watershed_map: dict[str, bool] = {}
        watershed_marker_threshold_map: dict[str, float | str] = {}
        watershed_selective_area_percentile_map: dict[str, float] = {}
        watershed_selective_elongation_threshold_map: dict[str, float] = {}
        for row in self.channel_rows:
            channel = row["channel"].get().strip().upper()
            if not channel or channel in ordered_channels:
                continue
            ordered_channels.append(channel)
            min_area_map[channel] = self._safe_int(row["min_area"].get(), 0)
            max_area_map[channel] = self._safe_int(row["max_area"].get(), 0)
            watershed_map[channel] = bool(row["watershed"].get())
            watershed_marker_threshold_map[channel] = self._parse_watershed_marker_threshold_value(
                row["watershed_marker_threshold_px"].get()
            )
            watershed_selective_area_percentile_map[channel] = float(
                np.clip(self._safe_float(row["watershed_selective_area_percentile"].get(), 90.0), 0.0, 100.0)
            )
            watershed_selective_elongation_threshold_map[channel] = max(
                1.0,
                self._safe_float(row["watershed_selective_elongation_threshold"].get(), 2.0),
            )
        return (
            ordered_channels,
            min_area_map,
            max_area_map,
            watershed_map,
            watershed_marker_threshold_map,
            watershed_selective_area_percentile_map,
            watershed_selective_elongation_threshold_map,
        )

    def _parse_watershed_marker_threshold_value(self, raw_value: str) -> float | str:
        stripped = str(raw_value).strip().lower()
        if stripped in {"", "auto"}:
            return "auto"
        return max(0.5, self._safe_float(stripped, 1.5))

    def _selected_overlap_rules(self) -> dict[str, dict[str, float | int | str]]:
        rules: dict[str, dict[str, float | int | str]] = {}
        for row in self.overlap_rule_rows:
            pair_key = self._pair_display_to_key(str(row["pair"].get()).strip().upper())
            if not pair_key:
                continue
            min_overlap_px = self._safe_int(row["min_overlap_px"].get(), 0)
            center_distance_px = self._safe_float(row["center_distance_px"].get(), 0.0)
            if min_overlap_px > 0 and center_distance_px > 0:
                rule = {
                    "method": "mask_overlap_then_centroid",
                    "overlap_threshold_px": min_overlap_px,
                    "fallback_distance_threshold_px": center_distance_px,
                }
            elif min_overlap_px > 0:
                rule = {
                    "method": "mask_overlap",
                    "overlap_threshold_px": min_overlap_px,
                }
            elif center_distance_px > 0:
                rule = {
                    "method": "centroid_distance",
                    "distance_threshold_px": center_distance_px,
                }
            else:
                rule = {}
            rules[pair_key] = dict(rule)
        return rules

    def _selected_overlap_sets(self) -> list[str]:
        combinations: list[str] = []
        for row in self.overlap_set_rows:
            combo = self._overlap_set_row_to_spec(row).upper()
            if combo:
                combinations.append(combo)
        return combinations

    def _infer_registration_channel(self, selected_channels: list[str], atlas_json_path: Path) -> str:
        if atlas_json_path.exists():
            try:
                registration = parse_registration_file(atlas_json_path)
                json_channels = []
                for reg_slice in registration.slices:
                    channel = extract_image_channel(reg_slice.filename).upper()
                    if channel and channel not in json_channels:
                        json_channels.append(channel)
                for channel in selected_channels:
                    if channel in json_channels:
                        return channel
                if json_channels:
                    return json_channels[0]
            except Exception:
                pass
        return selected_channels[0] if selected_channels else ""

    def _collect_config(self) -> AppConfig:
        config = self._base_config()
        image_folder = Path(self.image_folder_var.get().strip())
        segmentation_folder = Path(self.segmentation_folder_var.get().strip())
        atlas_json_path = Path(self.atlas_json_var.get().strip())
        output_folder = Path(self.output_folder_var.get().strip() or (Path.cwd() / "outputs"))
        sample_name = self.sample_name_var.get().strip() or "gui_run"

        (
            selected_channels,
            min_area_map,
            max_area_map,
            watershed_map,
            watershed_marker_threshold_map,
            watershed_selective_area_percentile_map,
            watershed_selective_elongation_threshold_map,
        ) = self._selected_channel_specs()
        overlap_rules = self._selected_overlap_rules()
        overlap_combinations = self._selected_overlap_sets()
        registration_channel = self._infer_registration_channel(selected_channels, atlas_json_path)

        config.discovery.image_folder = image_folder
        config.discovery.segmentation_folder = segmentation_folder
        config.discovery.atlas_json_path = atlas_json_path
        config.discovery.project_root = image_folder.parent if str(image_folder).strip() else config.discovery.project_root
        config.discovery.animal_include = []
        config.discovery.channel_include = []
        config.discovery.section_include = []

        config.output.output_root = output_folder
        config.output.run_name = sample_name

        config.processing.analysis_image_channels = selected_channels
        config.processing.normalize_ilastik_masks_to_binary = bool(self.normalize_ilastik_masks_var.get())
        config.processing.registration_image_channel = registration_channel
        config.processing.per_channel_min_area_px = {key: value for key, value in min_area_map.items() if value > 0}
        config.processing.per_channel_max_area_px = {key: value for key, value in max_area_map.items() if value > 0}
        config.processing.per_channel_apply_watershed = {key: value for key, value in watershed_map.items() if value}
        config.processing.per_channel_watershed_marker_threshold_px = {
            key: value for key, value in watershed_marker_threshold_map.items() if watershed_map.get(key, False)
        }
        config.processing.per_channel_watershed_selective_area_percentile = {
            key: value for key, value in watershed_selective_area_percentile_map.items() if watershed_map.get(key, False)
        }
        config.processing.per_channel_watershed_selective_elongation_threshold = {
            key: value for key, value in watershed_selective_elongation_threshold_map.items() if watershed_map.get(key, False)
        }
        config.processing.min_component_area_px = min(
            (value for value in min_area_map.values() if value > 0),
            default=config.processing.min_component_area_px,
        )
        config.processing.max_component_area_px = max(
            (value for value in max_area_map.values() if value > 0),
            default=0,
        )
        config.processing.border_assignment_policy = self.border_assignment_var.get().strip() or "bigger"
        config.processing.apply_watershed_to_masks = any(watershed_map.values())
        config.processing.watershed_marker_threshold_px = min(
            (
                value
                for key, value in watershed_marker_threshold_map.items()
                if watershed_map.get(key, False) and isinstance(value, (int, float))
            ),
            default="auto" if any(watershed_map.values()) else 1.5,
        )
        config.processing.watershed_selective_area_percentile = min(
            (
                value
                for key, value in watershed_selective_area_percentile_map.items()
                if watershed_map.get(key, False)
            ),
            default=90.0,
        )
        config.processing.watershed_selective_elongation_threshold = min(
            (
                value
                for key, value in watershed_selective_elongation_threshold_map.items()
                if watershed_map.get(key, False)
            ),
            default=2.0,
        )
        config.processing.parallel_workers = self._safe_int(self.parallel_workers_var.get(), 0)
        config.processing.overlay_enabled = True
        config.processing.combined_overlay_enabled = True
        config.processing.export_patch_ids = bool(self.export_patch_ids)

        config.matching.enabled = bool(overlap_combinations)
        config.matching.combinations = overlap_combinations
        config.matching.pair_rules = overlap_rules
        return config

    def _apply_config(self, config: AppConfig) -> None:
        self.image_folder_var.set("" if Path(config.discovery.image_folder) == Path(".") else str(config.discovery.image_folder))
        self.segmentation_folder_var.set(
            "" if Path(config.discovery.segmentation_folder) == Path(".") else str(config.discovery.segmentation_folder)
        )
        self.atlas_json_var.set("" if Path(config.discovery.atlas_json_path) == Path(".") else str(config.discovery.atlas_json_path))
        self.output_folder_var.set(str(config.output.output_root))
        self.sample_name_var.set(config.output.run_name or "gui_run")

        atlas_name = config.atlas.name or "allen_mouse"
        atlas_label_name = Path(config.atlas.labels_path).name if str(config.atlas.labels_path) else "unresolved"
        self.atlas_default_var.set(f"QUINTdeepflow atlas: {atlas_name} [{atlas_label_name}]")
        self.border_assignment_var.set(getattr(config.processing, "border_assignment_policy", "bigger") or "bigger")
        self.parallel_workers_var.set(str(int(getattr(config.processing, "parallel_workers", 0) or 0)))
        self.normalize_ilastik_masks_var.set(bool(getattr(config.processing, "normalize_ilastik_masks_to_binary", False)))
        channel_order = list(config.processing.analysis_image_channels)
        for channel_name in list(config.processing.per_channel_min_area_px) + list(config.processing.per_channel_max_area_px):
            if channel_name not in channel_order:
                channel_order.append(channel_name)
        self._set_channel_rows(
            channel_order,
            config.processing.per_channel_min_area_px,
            getattr(config.processing, "per_channel_max_area_px", {}),
            getattr(config.processing, "per_channel_apply_watershed", {}),
            getattr(config.processing, "per_channel_watershed_marker_threshold_px", {}),
            getattr(config.processing, "per_channel_watershed_selective_area_percentile", {}),
            getattr(config.processing, "per_channel_watershed_selective_elongation_threshold", {}),
            bool(getattr(config.processing, "apply_watershed_to_masks", False)),
            getattr(config.processing, "watershed_marker_threshold_px", 1.5),
            getattr(config.processing, "watershed_selective_area_percentile", 90.0),
            getattr(config.processing, "watershed_selective_elongation_threshold", 2.0),
        )
        self._set_overlap_rule_rows(config.matching.pair_rules)
        self._set_overlap_set_rows(config.matching.combinations)

    def _set_channel_rows(
        self,
        ordered_channels: list[str],
        min_area_map: dict[str, int],
        max_area_map: dict[str, int],
        watershed_map: dict[str, bool],
        watershed_marker_threshold_map: dict[str, float | str],
        watershed_selective_area_percentile_map: dict[str, float],
        watershed_selective_elongation_threshold_map: dict[str, float],
        global_watershed: bool = False,
        global_watershed_marker_threshold: float | str = 1.5,
        global_watershed_selective_area_percentile: float = 90.0,
        global_watershed_selective_elongation_threshold: float = 2.0,
    ) -> None:
        for row in self.channel_rows:
            row["channel"].set("")
            row["min_area"].set("")
            row["max_area"].set("")
            row["watershed"].set(False)
            row["watershed_marker_threshold_px"].set(str(global_watershed_marker_threshold))
            row["watershed_selective_area_percentile"].set(
                str(global_watershed_selective_area_percentile).rstrip("0").rstrip(".")
            )
            row["watershed_selective_elongation_threshold"].set(
                str(global_watershed_selective_elongation_threshold).rstrip("0").rstrip(".")
            )

        for index, channel in enumerate(ordered_channels[: len(self.channel_rows)]):
            row = self.channel_rows[index]
            row["channel"].set(channel.upper())
            min_area = min_area_map.get(channel.upper(), 0)
            max_area = max_area_map.get(channel.upper(), 0)
            row["min_area"].set(str(min_area) if min_area > 0 else "")
            row["max_area"].set(str(max_area) if max_area > 0 else "")
            channel_key = channel.upper()
            use_watershed = bool(watershed_map.get(channel_key, global_watershed))
            row["watershed"].set(use_watershed)
            marker_threshold = watershed_marker_threshold_map.get(channel_key, global_watershed_marker_threshold)
            row["watershed_marker_threshold_px"].set(str(marker_threshold))
            selective_area_percentile = watershed_selective_area_percentile_map.get(
                channel_key,
                global_watershed_selective_area_percentile,
            )
            row["watershed_selective_area_percentile"].set(
                str(float(selective_area_percentile)).rstrip("0").rstrip(".")
            )
            selective_elongation_threshold = watershed_selective_elongation_threshold_map.get(
                channel_key,
                global_watershed_selective_elongation_threshold,
            )
            row["watershed_selective_elongation_threshold"].set(
                str(float(selective_elongation_threshold)).rstrip("0").rstrip(".")
            )

    def _set_overlap_rule_rows(
        self,
        pair_rules: dict[str, dict[str, float | int | str]],
    ) -> None:
        ordered_pairs = sorted(
            [
                "+".join(sorted({channel.upper() for channel in OVERLAP_CHANNEL_PATTERN.findall(str(pair_key))}, key=str.upper))
                for pair_key in pair_rules
                if len({channel.upper() for channel in OVERLAP_CHANNEL_PATTERN.findall(str(pair_key))}) == 2
            ],
            key=str.upper,
        )
        self._ensure_overlap_rule_row_count(max(2, len(ordered_pairs)))
        for row in self.overlap_rule_rows:
            row["pair"].set("")
            row["min_overlap_px"].set("0")
            row["center_distance_px"].set("0")

        for index, pair_key in enumerate(ordered_pairs[: len(self.overlap_rule_rows)]):
            row = self.overlap_rule_rows[index]
            row["pair"].set(pair_key.replace("+", ", "))
            rule = pair_rules.get(pair_key, {})
            method = str(rule.get("method", ""))
            if method in {"mask_overlap", "mask_overlap_then_centroid", "mask_overlap_or_distance"}:
                row["min_overlap_px"].set(str(int(rule.get("overlap_threshold_px", 0))))
            if method == "centroid_distance":
                row["center_distance_px"].set(str(float(rule.get("distance_threshold_px", 0.0))))
            elif method in {"mask_overlap_then_centroid", "mask_overlap_or_distance"}:
                row["center_distance_px"].set(str(float(rule.get("fallback_distance_threshold_px", 0.0))))

    def _set_overlap_set_rows(self, combinations: list[str]) -> None:
        self._ensure_overlap_set_row_count(1)
        for row in self.overlap_set_rows:
            for index in range(1, 5):
                row[f"CH{index}"].set("")

        if combinations and self.overlap_set_rows:
            self._apply_overlap_set_spec_to_row(self.overlap_set_rows[0], str(combinations[0]).upper())

    def _validate_required_inputs(self) -> bool:
        missing: list[str] = []
        if not self.image_folder_var.get().strip():
            missing.append("Input raw image folder")
        if not self.segmentation_folder_var.get().strip():
            missing.append("ilastik segmentation folder")
        if not self.atlas_json_var.get().strip():
            missing.append(self.input_json_short_label)
        if not self.output_folder_var.get().strip():
            missing.append("Output folder")
        if not self.sample_name_var.get().strip():
            missing.append("Output sample name")
        selected_channels, _, _, _, _, _, _ = self._selected_channel_specs()
        if not selected_channels:
            missing.append("At least one analysis channel")
        if missing:
            messagebox.showerror("Missing input", "\n".join(missing))
            return False
        return True

    def _load_config_dialog(self) -> None:
        selected = filedialog.askopenfilename(
            initialdir=str(self.current_config_path.parent),
            filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
        )
        if not selected:
            return
        self.current_config_path = Path(selected)
        self._apply_config(load_app_config(self.current_config_path))
        self._append_log(f"Loaded config: {self.current_config_path}")

    def _save_config_dialog(self) -> None:
        if not self._validate_required_inputs():
            return
        selected = filedialog.asksaveasfilename(
            initialdir=str(self.current_config_path.parent),
            defaultextension=".yaml",
            filetypes=[("YAML", "*.yaml *.yml"), ("All", "*.*")],
        )
        if not selected:
            return
        self.current_config_path = Path(selected)
        save_gui_visible_app_config(self._collect_config(), self.current_config_path)
        self._append_log(f"Saved config: {self.current_config_path}")

    def _discover(self) -> None:
        if not self._validate_required_inputs():
            return
        try:
            groups = discover_section_groups(self._collect_config())
            frame = discovery_to_dataframe(groups)
        except Exception as exc:
            messagebox.showerror(f"{self.input_json_short_label} check failed", str(exc))
            self._append_log(f"{self.input_json_short_label} check failed: {exc}")
            return
        self._populate_tree(frame)
        self._append_log(f"{self.input_json_short_label} check found {len(frame)} bundle(s)")

    def _run_pipeline(self) -> None:
        if not self._validate_required_inputs():
            return
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showinfo("Run in progress", f"{self.app_title} is already processing this dataset.")
            return
        config = self._collect_config()
        output_dir = config.output.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        auto_config_path = output_dir / "gui_run_config.yaml"
        save_gui_visible_app_config(config, auto_config_path)
        self._append_log(f"Auto-saved run config: {auto_config_path}")
        self.status_var.set("Running...")
        self.progress.configure(value=0)
        self._run_in_progress = True
        self._active_output_dir = output_dir
        self._worker_thread = threading.Thread(target=self._pipeline_worker, args=(config,), daemon=False)
        self._worker_thread.start()

    def _pipeline_worker(self, config: AppConfig) -> None:
        try:
            pipeline = QuantificationPipeline(config)
            result = pipeline.run(progress_callback=lambda msg, pct: self.queue.put(("progress", (msg, pct))))
            self.queue.put(("done", result))
        except Exception as exc:
            self._write_worker_error_log(exc)
            self.queue.put(("error", str(exc)))

    def _poll_queue(self) -> None:
        try:
            while True:
                event, payload = self.queue.get_nowait()
                if event == "progress":
                    message, fraction = payload
                    self.status_var.set(str(message))
                    self.progress.configure(value=max(0, min(100, int(float(fraction) * 100))))
                    self._append_log(str(message))
                elif event == "done":
                    result = payload
                    self._run_in_progress = False
                    self._worker_thread = None
                    self.status_var.set("Completed")
                    self.progress.configure(value=100)
                    self.result_output_dir = result.output_dir
                    self._active_output_dir = result.output_dir
                    self._append_log(f"Run complete: {result.output_dir}")
                    self._populate_tree(result.discovery_table)
                elif event == "error":
                    self._run_in_progress = False
                    self._worker_thread = None
                    self.status_var.set("Error")
                    self._append_log(f"Run failed: {payload}")
                    messagebox.showerror("Pipeline failed", str(payload))
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)

    def _write_worker_error_log(self, exc: Exception) -> None:
        if self._active_output_dir is None:
            return
        try:
            import traceback

            log_path = self._active_output_dir / "qdf2_gui_worker_error.log"
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
        except Exception:
            pass

    def _on_close_request(self) -> None:
        if self._worker_thread and self._worker_thread.is_alive():
            keep_running = messagebox.askyesno(
                "Processing in progress",
                "Quantification is still running.\n\n"
                f"If you close this window now, {self.app_title} will keep processing in the background "
                "until cell_level.csv and the other output files are finished.\n\n"
                "Close the window anyway?",
            )
            if not keep_running:
                return
        self.destroy()

    def _populate_tree(self, frame) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        if frame is None or frame.empty:
            return
        for row in frame.to_dict("records"):
            values = (
                row.get("animal_id", ""),
                row.get("section_id", ""),
                row.get("image_channel", "") or row.get("channel", ""),
                row.get("image_file", ""),
                row.get("mask_file", ""),
                row.get("registration_json", ""),
                row.get("registration_entry", ""),
            )
            self.tree.insert("", "end", values=values)

    def _append_log(self, message: str) -> None:
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def _open_results(self) -> None:
        target = self.result_output_dir
        if target is None:
            candidate = self._collect_config().output.output_dir
            if candidate.exists():
                target = candidate
        if target and target.exists():
            os.startfile(target)  # type: ignore[attr-defined]
        else:
            messagebox.showinfo("Results", "No result folder available yet.")

    def _export_portable_bundle(self) -> None:
        destination = filedialog.askdirectory(initialdir=str(Path.cwd()))
        if not destination:
            return
        try:
            source_root = Path(__file__).resolve().parents[2]
            bundle_root = export_portable_bundle(source_root, Path(destination))
        except Exception as exc:
            self._append_log(f"Portable export failed: {exc}")
            messagebox.showerror("Portable export failed", str(exc))
            return
        self._append_log(f"Portable bundle exported: {bundle_root}")
        messagebox.showinfo("Portable export", f"Portable bundle written to:\n{bundle_root}")

def launch_gui() -> None:
    """Start the Tkinter app."""

    app = QuintDeepflowApp()
    app.mainloop()
