"""Tkinter GUI for QUINTdeepflow DeepSlice preparation and execution."""

from __future__ import annotations

import logging
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from app_version import APP_VERSION, version_label
from deepslice.pipeline import normalize_ap_hint_mode, resolve_deepslice_python, run_deepslice_job
from io_utils.portable_bundle import export_portable_bundle


class DeepSliceApp(tk.Tk):
    """Small Windows-friendly GUI for QUINTdeepflow DeepSlice execution."""

    WINDOW_GEOMETRY = "980x620"
    WINDOW_MINSIZE = (900, 560)

    def __init__(self) -> None:
        super().__init__()
        self.withdraw()
        self.title(version_label("QUINTdeepflow1"))
        self.geometry(self.WINDOW_GEOMETRY)
        self.minsize(*self.WINDOW_MINSIZE)
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.result_jpg_dir: Path | None = None

        self.input_dir_var = tk.StringVar()
        self.channel_var = tk.StringVar(value="CH3")
        self.same_slicing_angle_var = tk.BooleanVar(value=False)
        self.ap_hint_path_var = tk.StringVar()
        self.ap_hint_mode_var = tk.StringVar(value="Disabled")
        self.status_var = tk.StringVar(value="Ready")
        self.deepslice_python_var = tk.StringVar(value=self._resolve_deepslice_python_label())

        self._configure_style()
        self._build_layout()
        self._stabilize_window()
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

    def _stabilize_window(self) -> None:
        self.update_idletasks()
        self.geometry(self.WINDOW_GEOMETRY)
        self.minsize(*self.WINDOW_MINSIZE)

    def _resolve_deepslice_python_label(self) -> str:
        try:
            return str(resolve_deepslice_python())
        except Exception:
            return "DeepSlice Python not found"

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        form = ttk.LabelFrame(self, text="QUINTdeepflow DeepSlice")
        form.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        form.columnconfigure(0, minsize=240)
        form.columnconfigure(1, weight=1)
        form.columnconfigure(2, minsize=94)

        ttk.Label(form, text="1. Target directory with TIFF files").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(form, textvariable=self.input_dir_var).grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=6)
        ttk.Button(form, text="Browse", command=self._browse_input_dir).grid(row=0, column=2, sticky="ew", padx=(0, 6), pady=6)

        ttk.Label(form, text="2. Channel for DeepSlice").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            form,
            textvariable=self.channel_var,
            values=["CH1", "CH2", "CH3", "CH4"],
            state="readonly",
            width=10,
        ).grid(row=1, column=1, sticky="w", padx=(0, 6), pady=6)

        ttk.Checkbutton(
            form,
            text="3. Same slicing angle (ABBA Allow change of atlas slicing angle)",
            variable=self.same_slicing_angle_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=6)

        ttk.Label(form, text="4. AP hint file (optional)").grid(row=3, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(form, textvariable=self.ap_hint_path_var).grid(row=3, column=1, sticky="ew", padx=(0, 6), pady=6)
        ttk.Button(form, text="Browse", command=self._browse_ap_hint_file).grid(row=3, column=2, sticky="ew", padx=(0, 6), pady=6)

        ttk.Label(form, text="5. AP hint handling").grid(row=4, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            form,
            textvariable=self.ap_hint_mode_var,
            values=["Disabled", "Rerun if AP deviates"],
            state="readonly",
            width=30,
        ).grid(row=4, column=1, sticky="w", padx=(0, 6), pady=6)

        ttk.Label(form, text="DeepSlice Python").grid(row=5, column=0, sticky="w", padx=6, pady=6)
        ttk.Label(form, textvariable=self.deepslice_python_var, foreground="#375a7f").grid(
            row=5,
            column=1,
            columnspan=2,
            sticky="w",
            padx=(0, 6),
            pady=6,
        )

        note = ttk.Label(
            form,
            text="Run creates a sibling 'jpg' folder, writes same-size compact JPEGs, "
            "runs enhanced DeepSlice inference, and saves jpgDS_results.json/.csv/.xml into that folder. "
            "AP hint file format: 1st column = slice_id_or_filename (XY01 or full filename), "
            "2nd = ap_mm, 3rd = tolerance_mm. Example: XY01,2.40,0.30. "
            "'Rerun if AP deviates' uses AP hint order first, then rewrites only out-of-range slices to the hint AP.",
            foreground="#444444",
            wraplength=900,
            justify="left",
        )
        note.grid(row=6, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 6))

        button_frame = ttk.Frame(form)
        button_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        for text, command in (
            ("Run", self._run),
            ("Open jpg folder", self._open_jpg_folder),
            ("Export Portable", self._export_portable_bundle),
        ):
            ttk.Button(button_frame, text=text, command=command, width=18).pack(side="left", padx=4)

        body = ttk.LabelFrame(self, text="Log")
        body.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.log_text = ScrolledText(body, wrap="word", font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky="nsew")

        status = ttk.Frame(self)
        status.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        status.columnconfigure(1, weight=1)
        status.columnconfigure(2, weight=0)
        ttk.Label(status, textvariable=self.status_var, width=34, anchor="w").grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 8),
        )
        self.progress = ttk.Progressbar(status, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=1, sticky="ew")
        ttk.Label(status, text=f"Version {APP_VERSION}", anchor="e", foreground="#5f6b7a").grid(
            row=0,
            column=2,
            sticky="e",
            padx=(8, 0),
        )

    def _browse_input_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.input_dir_var.get() or os.getcwd())
        if selected:
            self.input_dir_var.set(selected)

    def _browse_ap_hint_file(self) -> None:
        selected = filedialog.askopenfilename(
            initialdir=self.ap_hint_path_var.get() or self.input_dir_var.get() or os.getcwd(),
            filetypes=[
                ("Hint files", "*.csv *.tsv *.txt *.xlsx *.xls"),
                ("CSV", "*.csv"),
                ("Excel", "*.xlsx *.xls"),
                ("All files", "*.*"),
            ],
        )
        if selected:
            self.ap_hint_path_var.set(selected)

    def _run(self) -> None:
        input_dir = Path(self.input_dir_var.get().strip())
        if not str(input_dir).strip():
            messagebox.showerror("Missing input", "Target directory is required.")
            return

        ap_hint_text = self.ap_hint_path_var.get().strip()
        ap_hint_path = Path(ap_hint_text) if ap_hint_text else None
        if ap_hint_path is not None and not ap_hint_path.exists():
            messagebox.showerror("Missing AP hint file", f"AP hint file not found:\n{ap_hint_path}")
            return

        self.progress["value"] = 0
        self.status_var.set("Running...")
        self._append_log(f"Starting DeepSlice for {self.channel_var.get().strip().upper()} in {input_dir}")
        self._append_log(
            "Options: same_slicing_angle=%s, ap_hint_mode=%s%s"
            % (
                bool(self.same_slicing_angle_var.get()),
                normalize_ap_hint_mode(self.ap_hint_mode_var.get()),
                f", ap_hint_file={ap_hint_path}" if ap_hint_path is not None else "",
            )
        )

        def worker() -> None:
            logger = _QueueLogger(self.queue)
            try:
                result = run_deepslice_job(
                    input_dir=input_dir,
                    channel=self.channel_var.get().strip().upper(),
                    progress_callback=lambda message, fraction: self.queue.put(("progress", (message, fraction))),
                    logger=logger,
                    same_slicing_angle=bool(self.same_slicing_angle_var.get()),
                    ap_hint_path=ap_hint_path,
                    ap_hint_mode=self.ap_hint_mode_var.get(),
                )
                self.queue.put(("done", result))
            except Exception as exc:
                self.queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _open_jpg_folder(self) -> None:
        if self.result_jpg_dir is None:
            base_dir = Path(self.input_dir_var.get().strip())
            if not str(base_dir).strip():
                return
            candidate = base_dir.parent / "jpg"
        else:
            candidate = self.result_jpg_dir
        if candidate.exists():
            os.startfile(candidate)  # type: ignore[attr-defined]

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")

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

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "progress":
                    message, fraction = payload  # type: ignore[misc]
                    self.status_var.set(str(message))
                    self.progress["value"] = max(0, min(int(float(fraction) * 100), 100))
                    self._append_log(str(message))
                elif kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    result = payload
                    self.result_jpg_dir = result.jpg_dir
                    self.status_var.set("Done")
                    self.progress["value"] = 100
                    self._append_log(f"DeepSlice complete: {result.json_path}")
                    self._append_log(f"JPEG folder: {result.jpg_dir}")
                    if result.ap_hint_report_path is not None:
                        self._append_log(f"AP hint report: {result.ap_hint_report_path}")
                    if result.reran_with_ap_hint_order:
                        self._append_log("AP-guided slice order was applied.")
                elif kind == "error":
                    self.status_var.set("Failed")
                    self.progress["value"] = 0
                    self._append_log(f"Error: {payload}")
                    messagebox.showerror("DeepSlice failed", str(payload))
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)


class _QueueLogger(logging.Logger):
    """Minimal logger that forwards messages into the GUI queue."""

    def __init__(self, event_queue: queue.Queue[tuple[str, object]]) -> None:
        super().__init__("quint_deepslice_gui")
        self.event_queue = event_queue

    def info(self, msg, *args, **kwargs) -> None:  # type: ignore[override]
        text = str(msg) % args if args else str(msg)
        self.event_queue.put(("log", text))

    def warning(self, msg, *args, **kwargs) -> None:  # type: ignore[override]
        text = str(msg) % args if args else str(msg)
        self.event_queue.put(("log", f"WARNING: {text}"))
