"""Tkinter GUI for QDFevo_1_Align AP-constrained DeepSlice."""

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
from deepslice.evo_refine import run_qdf1_evo_job
from deepslice.pipeline import resolve_deepslice_python
from io_utils.portable_bundle import export_portable_bundle


class DeepSliceEvoApp(tk.Tk):
    """Windows-friendly GUI for QDFevo_1_Align registration."""

    WINDOW_GEOMETRY = "1040x690"
    WINDOW_MINSIZE = (940, 620)

    def __init__(self) -> None:
        super().__init__()
        self.withdraw()
        self.title(version_label("QDFevo_1_Align"))
        self.geometry(self.WINDOW_GEOMETRY)
        self.minsize(*self.WINDOW_MINSIZE)
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.result_jpg_dir: Path | None = None

        self.input_dir_var = tk.StringVar()
        self.channel_var = tk.StringVar(value="CH3")
        self.same_slicing_angle_var = tk.BooleanVar(value=True)
        self.ap_hint_path_var = tk.StringVar()
        self.constrain_to_ap_hints_var = tk.BooleanVar(value=True)
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

        form = ttk.LabelFrame(self, text="QDFevo_1_Align: AP-limited DeepSlice")
        form.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        form.columnconfigure(0, minsize=250)
        form.columnconfigure(1, weight=1)
        form.columnconfigure(2, minsize=94)

        ttk.Label(form, text="1. Target directory with TIFF files").grid(row=0, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(form, textvariable=self.input_dir_var).grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=5)
        ttk.Button(form, text="Browse", command=self._browse_input_dir).grid(row=0, column=2, sticky="ew", padx=(0, 6), pady=5)

        ttk.Label(form, text="2. Channel for fitting").grid(row=1, column=0, sticky="w", padx=6, pady=5)
        ttk.Combobox(
            form,
            textvariable=self.channel_var,
            values=["CH1", "CH2", "CH3", "CH4"],
            state="readonly",
            width=10,
        ).grid(row=1, column=1, sticky="w", padx=(0, 6), pady=5)

        ttk.Checkbutton(
            form,
            text="3. Use one consistent slicing-angle model during DeepSlice",
            variable=self.same_slicing_angle_var,
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=6, pady=5)

        ttk.Label(form, text="4. AP hint file").grid(row=3, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(form, textvariable=self.ap_hint_path_var).grid(row=3, column=1, sticky="ew", padx=(0, 6), pady=5)
        ttk.Button(form, text="Browse", command=self._browse_ap_hint_file).grid(row=3, column=2, sticky="ew", padx=(0, 6), pady=5)

        ttk.Checkbutton(
            form,
            text="5. Constrain DeepSlice AP estimates to the AP hint range",
            variable=self.constrain_to_ap_hints_var,
        ).grid(row=4, column=0, columnspan=3, sticky="w", padx=6, pady=5)

        ttk.Label(form, text="DeepSlice Python").grid(row=5, column=0, sticky="w", padx=6, pady=5)
        ttk.Label(form, textvariable=self.deepslice_python_var, foreground="#375a7f").grid(
            row=5,
            column=1,
            columnspan=2,
            sticky="w",
            padx=(0, 6),
            pady=5,
        )

        note = ttk.Label(
            form,
            text=(
                "Output: QDFevo_1_Align writes QDFevo1_results.json/.csv/.xml from DeepSlice predictions constrained by AP hints. "
                "AP hint file columns: filename or section_id, ap_mm, tolerance_mm. "
                "No cleanup, edge refinement, roll correction, pan shift, or scale correction is applied in this default workflow."
            ),
            foreground="#444444",
            wraplength=960,
            justify="left",
        )
        note.grid(row=6, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 6))

        button_frame = ttk.Frame(form)
        button_frame.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        for text, command in (
            ("Run QDFevo_1", self._run),
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
        ttk.Label(status, textvariable=self.status_var, width=40, anchor="w").grid(row=0, column=0, sticky="w", padx=(0, 8))
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
        self._append_log(f"Starting QDFevo_1_Align for {self.channel_var.get().strip().upper()} in {input_dir}")
        self._append_log(
            "Options: same_slicing_angle=%s, constrain_to_ap_hints=%s%s"
            % (
                bool(self.same_slicing_angle_var.get()),
                bool(self.constrain_to_ap_hints_var.get()),
                f", ap_hint_file={ap_hint_path}" if ap_hint_path is not None else "",
            )
        )

        def worker() -> None:
            logger = _QueueLogger(self.queue)
            try:
                result = run_qdf1_evo_job(
                    input_dir=input_dir,
                    channel=self.channel_var.get().strip().upper(),
                    progress_callback=lambda message, fraction: self.queue.put(("progress", (message, fraction))),
                    logger=logger,
                    same_slicing_angle=bool(self.same_slicing_angle_var.get()),
                    ap_hint_path=ap_hint_path,
                    constrain_to_ap_hints=bool(self.constrain_to_ap_hints_var.get()),
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

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "progress":
                    message, fraction = payload  # type: ignore[misc]
                    self.status_var.set(str(message))
                    self.progress["value"] = int(float(fraction) * 100)
                elif kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    result = payload
                    self.result_jpg_dir = result.jpg_dir  # type: ignore[attr-defined]
                    self.status_var.set("Done")
                    self.progress["value"] = 100
                    self._append_log(f"Output JSON: {result.refined_json_path}")  # type: ignore[attr-defined]
                    self._append_log(f"Report: {result.report_path}")  # type: ignore[attr-defined]
                    messagebox.showinfo("QDFevo_1_Align finished", f"Output JSON:\n{result.refined_json_path}")  # type: ignore[attr-defined]
                elif kind == "error":
                    self.status_var.set("Failed")
                    self._append_log(f"ERROR: {payload}")
                    messagebox.showerror("QDFevo_1_Align failed", str(payload))
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)


class _QueueLogger(logging.Logger):
    """Logger that forwards messages into the GUI queue."""

    def __init__(self, target_queue: queue.Queue[tuple[str, object]]) -> None:
        super().__init__("QDFevo1AlignQueue")
        self.target_queue = target_queue

    def info(self, msg: object, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        self.target_queue.put(("log", str(msg) % args if args else str(msg)))

    def warning(self, msg: object, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        self.target_queue.put(("log", "WARNING: " + (str(msg) % args if args else str(msg))))

    def error(self, msg: object, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        self.target_queue.put(("log", "ERROR: " + (str(msg) % args if args else str(msg))))


def main() -> None:
    app = DeepSliceEvoApp()
    app.mainloop()


if __name__ == "__main__":
    main()
