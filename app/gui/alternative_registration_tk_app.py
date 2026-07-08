"""Tkinter GUI for experimental QDF1 registration adapters."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from app_version import APP_VERSION, version_label
from registration_adapters.alternative_pipeline import (
    AlternativeRegistrationResult,
    run_ambia_adapter,
)


class AlternativeRegistrationApp(tk.Tk):
    """GUI wrapper for the AMBIA QDF1 experiment."""

    WINDOW_GEOMETRY = "1040x660"
    WINDOW_MINSIZE = (940, 600)

    def __init__(self, method: str) -> None:
        super().__init__()
        self.withdraw()
        self.method = "AMBIA"
        self.title(version_label("QUINTdeepflow1-AMBIA"))
        self.geometry(self.WINDOW_GEOMETRY)
        self.minsize(*self.WINDOW_MINSIZE)
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.result_output_dir: Path | None = None

        self.input_dir_var = tk.StringVar(value="")
        self.registration_channel_var = tk.StringVar(value="CH3")
        self.secondary_channel_var = tk.StringVar(value="")
        self.output_dir_var = tk.StringVar()
        self.ambia_root_var = tk.StringVar(value=self._default_ambia_root())
        self.ap_hint_path_var = tk.StringVar()
        self.enforce_ap_order_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")

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

    def _default_ambia_root(self) -> str:
        candidates = [
            Path.home() / "AMBIA" / "AMBIA",
            Path(__file__).resolve().parents[1] / "external_methods" / "AMBIA",
        ]
        for candidate in candidates:
            if (candidate / "Gui_Atlases" / "Adult_full_atlases").exists():
                return str(candidate)
        return ""

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        form = ttk.LabelFrame(self, text="QDF1-AMBIA adapter")
        form.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        form.columnconfigure(0, minsize=240)
        form.columnconfigure(1, weight=1)
        form.columnconfigure(2, minsize=94)

        ttk.Label(form, text="1. Raw image folder").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(form, textvariable=self.input_dir_var).grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=6)
        ttk.Button(form, text="Browse", command=self._browse_input_dir).grid(row=0, column=2, sticky="ew", padx=(0, 6), pady=6)

        ttk.Label(form, text="2. Registration channel").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            form,
            textvariable=self.registration_channel_var,
            values=["CH1", "CH2", "CH3", "CH4"],
            state="readonly",
            width=10,
        ).grid(row=1, column=1, sticky="w", padx=(0, 6), pady=6)

        ttk.Label(form, text="3. Optional second channel").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            form,
            textvariable=self.secondary_channel_var,
            values=["", "CH1", "CH2", "CH3", "CH4"],
            state="readonly",
            width=10,
        ).grid(row=2, column=1, sticky="w", padx=(0, 6), pady=6)

        ttk.Label(form, text="4. Output folder").grid(row=3, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(form, textvariable=self.output_dir_var).grid(row=3, column=1, sticky="ew", padx=(0, 6), pady=6)
        ttk.Button(form, text="Browse", command=self._browse_output_dir).grid(row=3, column=2, sticky="ew", padx=(0, 6), pady=6)

        next_row = 4
        ttk.Label(form, text="5. AMBIA repository folder").grid(row=next_row, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(form, textvariable=self.ambia_root_var).grid(row=next_row, column=1, sticky="ew", padx=(0, 6), pady=6)
        ttk.Button(form, text="Browse", command=self._browse_ambia_root).grid(row=next_row, column=2, sticky="ew", padx=(0, 6), pady=6)
        next_row += 1
        ttk.Label(form, text="6. AP hint file (optional)").grid(row=next_row, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(form, textvariable=self.ap_hint_path_var).grid(row=next_row, column=1, sticky="ew", padx=(0, 6), pady=6)
        ttk.Button(form, text="Browse", command=self._browse_ap_hint).grid(row=next_row, column=2, sticky="ew", padx=(0, 6), pady=6)
        next_row += 1
        ttk.Checkbutton(
            form,
            text="Keep section order while matching AP position",
            variable=self.enforce_ap_order_var,
        ).grid(row=next_row, column=1, sticky="w", padx=(0, 6), pady=(0, 6))
        next_row += 1
        note_text = (
            "Run prepares QDF1 JPEGs, matches each section against AMBIA's local atlas stack, "
            "and writes a DeepSlice/QuickNII-compatible JSON under the jpg folder. "
            "AP hints can restrict the allowed AP range when rough slice positions are available."
        )

        note = ttk.Label(form, text=note_text, foreground="#444444", wraplength=960, justify="left")
        note.grid(row=next_row, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 6))
        next_row += 1

        button_frame = ttk.Frame(form)
        button_frame.grid(row=next_row, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Button(button_frame, text="Run", command=self._run, width=18).pack(side="left", padx=4)
        ttk.Button(button_frame, text="Open output", command=self._open_output, width=18).pack(side="left", padx=4)

        body = ttk.LabelFrame(self, text="Log")
        body.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.log_text = ScrolledText(body, wrap="word", font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky="nsew")

        status = ttk.Frame(self)
        status.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        status.columnconfigure(1, weight=1)
        ttk.Label(status, textvariable=self.status_var, width=36, anchor="w").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.progress = ttk.Progressbar(status, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=1, sticky="ew")
        ttk.Label(status, text=f"Version {APP_VERSION}", anchor="e", foreground="#5f6b7a").grid(row=0, column=2, sticky="e", padx=(8, 0))

    def _browse_input_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.input_dir_var.get() or os.getcwd())
        if selected:
            self.input_dir_var.set(selected)

    def _browse_output_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.output_dir_var.get() or self.input_dir_var.get() or os.getcwd())
        if selected:
            self.output_dir_var.set(selected)

    def _browse_ambia_root(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.ambia_root_var.get() or os.getcwd())
        if selected:
            self.ambia_root_var.set(selected)

    def _browse_ap_hint(self) -> None:
        selected = filedialog.askopenfilename(
            initialdir=self.input_dir_var.get() or os.getcwd(),
            filetypes=[("Hint files", "*.csv *.tsv *.txt *.xlsx *.xls"), ("All files", "*.*")],
        )
        if selected:
            self.ap_hint_path_var.set(selected)

    def _run(self) -> None:
        input_dir = Path(self.input_dir_var.get().strip())
        if not str(input_dir).strip():
            messagebox.showerror("Missing input", "Raw image folder is required.")
            return
        output_dir_text = self.output_dir_var.get().strip()
        output_dir = Path(output_dir_text) if output_dir_text else None
        self.progress["value"] = 0
        self.status_var.set("Running...")
        self._append_log(f"Starting {self.method} adapter for {self.registration_channel_var.get()} in {input_dir}")

        def worker() -> None:
            logger = _QueueLogger(self.queue)
            try:
                ap_hint_text = self.ap_hint_path_var.get().strip()
                result = run_ambia_adapter(
                    input_dir=input_dir,
                    channel=self.registration_channel_var.get(),
                    output_dir=output_dir,
                    secondary_channel=self.secondary_channel_var.get(),
                    ambia_root=Path(self.ambia_root_var.get().strip()) if self.ambia_root_var.get().strip() else None,
                    ap_hint_path=Path(ap_hint_text) if ap_hint_text else None,
                    enforce_ap_order=bool(self.enforce_ap_order_var.get()),
                    progress_callback=lambda message, fraction: self.queue.put(("progress", (message, fraction))),
                    logger=logger,
                )
                self.queue.put(("done", result))
            except Exception as exc:
                self.queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _open_output(self) -> None:
        candidate = self.result_output_dir
        if candidate is None:
            output_text = self.output_dir_var.get().strip()
            candidate = Path(output_text) if output_text else None
        if candidate and candidate.exists():
            os.startfile(candidate)  # type: ignore[attr-defined]

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
                    self.progress["value"] = max(0, min(int(float(fraction) * 100), 100))
                    self._append_log(str(message))
                elif kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    result: AlternativeRegistrationResult = payload  # type: ignore[assignment]
                    self.result_output_dir = result.output_dir
                    self.status_var.set(result.status)
                    self.progress["value"] = 100
                    self._append_log(f"Status: {result.status}")
                    self._append_log(f"Output: {result.output_dir}")
                    self._append_log(f"Report: {result.report_path}")
                    if result.qdf_json_path is not None:
                        self._append_log(f"QDF1 JSON: {result.qdf_json_path}")
                elif kind == "error":
                    self.status_var.set("Failed")
                    self.progress["value"] = 0
                    self._append_log(f"Error: {payload}")
                    messagebox.showerror(f"{self.method} adapter failed", str(payload))
        except queue.Empty:
            pass
        self.after(150, self._poll_queue)


class _QueueLogger(logging.Logger):
    """Minimal logger that forwards messages into the GUI queue."""

    def __init__(self, event_queue: queue.Queue[tuple[str, object]]) -> None:
        super().__init__("quint_alternative_registration_gui")
        self.event_queue = event_queue

    def info(self, msg, *args, **kwargs) -> None:  # type: ignore[override]
        text = str(msg) % args if args else str(msg)
        self.event_queue.put(("log", text))

    def warning(self, msg, *args, **kwargs) -> None:  # type: ignore[override]
        text = str(msg) % args if args else str(msg)
        self.event_queue.put(("log", f"WARNING: {text}"))
