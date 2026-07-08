"""Tkinter GUI for QUINTdeepflow3 omit-region quality control."""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import ImageTk

from app_version import APP_VERSION, version_label
from quality_check.core import (
    build_omit_rows,
    create_default_session,
    export_omit_outputs,
    load_overlay_dataset,
    load_overlay_stack,
    load_session,
    normalize_session,
    region_info_for_code,
    save_session,
    session_path_for_overlay_dir,
)
from quality_check.models import OmitRegionSelection, OmitSessionState, OverlayChannelInfo, OverlayDataset, OverlaySliceInfo
from quality_check.render import component_selection_at_xy, compute_canvas_transform, display_code_at_xy, render_qc_image


LOG = logging.getLogger(__name__)


class QuintDeepflow3App(tk.Tk):
    """Interactive omit-region quality-check GUI."""

    WINDOW_GEOMETRY = "1580x940"
    WINDOW_MINSIZE = (1320, 820)
    ZOOM_CHOICES = ("Fit", "50%", "75%", "100%", "150%", "200%", "300%", "400%")

    def __init__(self) -> None:
        super().__init__()
        self.withdraw()
        self.title(version_label("QUINTdeepflow3"))
        self.geometry(self.WINDOW_GEOMETRY)
        self.minsize(*self.WINDOW_MINSIZE)

        self.overlay_dir_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")
        self.hover_var = tk.StringVar(value="Hover region: -")
        self.slice_summary_var = tk.StringVar(value="No overlay loaded")
        self.zoom_var = tk.StringVar(value="Fit")

        self.dataset: OverlayDataset | None = None
        self.session: OmitSessionState | None = None
        self.current_slice: OverlaySliceInfo | None = None
        self.current_stack = None
        self.current_transform = None
        self.channel_visibility: dict[str, tk.BooleanVar] = {}
        self.channel_checkbuttons: list[ttk.Checkbutton] = []
        self._photo = None
        self._last_mouse_canvas_xy: tuple[float, float] | None = None
        self._pending_center_image_xy: tuple[float, float] | None = None

        self._configure_style()
        self._build_layout()
        self._stabilize_window()
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

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.LabelFrame(self, text="QUINTdeepflow3 Quality Check")
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Overlay directory").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(top, textvariable=self.overlay_dir_var).grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=6)
        ttk.Button(top, text="Browse", command=self._browse_overlay_dir, width=10).grid(row=0, column=2, sticky="ew", padx=(0, 6), pady=6)
        ttk.Button(top, text="Load", command=self._load_overlay_dir, width=10).grid(row=0, column=3, sticky="ew", padx=(0, 6), pady=6)
        ttk.Button(top, text="Load Session", command=self._load_session_dialog, width=12).grid(row=0, column=4, sticky="ew", padx=(0, 6), pady=6)
        ttk.Button(top, text="Save Session", command=self._save_session_dialog, width=12).grid(row=0, column=5, sticky="ew", padx=(0, 6), pady=6)
        ttk.Button(top, text="Finish", command=self._finish, width=12).grid(row=0, column=6, sticky="ew", padx=(0, 6), pady=6)

        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        left = ttk.Frame(paned)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        paned.add(left, weight=3)

        right = ttk.Frame(paned, width=420)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        paned.add(right, weight=1)

        toolbar = ttk.Frame(left)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(toolbar, text="Zoom").pack(side="left", padx=(0, 6))
        self.zoom_combo = ttk.Combobox(
            toolbar,
            textvariable=self.zoom_var,
            values=self.ZOOM_CHOICES,
            width=8,
            state="readonly",
        )
        self.zoom_combo.pack(side="left")
        self.zoom_combo.bind("<<ComboboxSelected>>", self._on_zoom_combo_selected)
        ttk.Button(toolbar, text="Fit", command=self._set_zoom_fit, width=6).pack(side="left", padx=(8, 4))
        ttk.Button(toolbar, text="-", command=lambda: self._step_zoom(-1), width=4).pack(side="left", padx=2)
        ttk.Button(toolbar, text="+", command=lambda: self._step_zoom(1), width=4).pack(side="left", padx=2)

        canvas_frame = ttk.Frame(left)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, background="#202020", highlightthickness=0, cursor="crosshair")
        self.h_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal", command=self.canvas.xview)
        self.v_scroll = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        self.canvas.bind("<Configure>", lambda _event: self._render_current_slice())
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_hover)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)

        ttk.Label(right, textvariable=self.slice_summary_var, wraplength=360, justify="left").grid(row=0, column=0, sticky="ew", padx=6, pady=(0, 8))

        notebook = ttk.Notebook(right)
        notebook.grid(row=1, column=0, sticky="nsew")

        slice_tab = ttk.Frame(notebook)
        channel_tab = ttk.Frame(notebook)
        omit_tab = ttk.Frame(notebook)
        notebook.add(slice_tab, text="Slices")
        notebook.add(channel_tab, text="Channels")
        notebook.add(omit_tab, text="Omit list")

        slice_tab.columnconfigure(0, weight=1)
        slice_tab.rowconfigure(0, weight=1)
        self.slice_listbox = tk.Listbox(slice_tab, exportselection=False, activestyle="dotbox")
        slice_scroll = ttk.Scrollbar(slice_tab, orient="vertical", command=self.slice_listbox.yview)
        self.slice_listbox.configure(yscrollcommand=slice_scroll.set)
        self.slice_listbox.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        slice_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 6), pady=6)
        self.slice_listbox.bind("<<ListboxSelect>>", self._on_slice_select)

        channel_tab.columnconfigure(0, weight=1)
        channel_tab.rowconfigure(0, weight=1)
        self.channel_frame = ttk.Frame(channel_tab)
        self.channel_frame.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        omit_tab.columnconfigure(0, weight=1)
        omit_tab.rowconfigure(0, weight=1)
        self.omit_tree = ttk.Treeview(omit_tab, columns=("code", "patch", "region", "hemisphere"), show="headings", height=10)
        self.omit_tree.heading("code", text="Display code")
        self.omit_tree.heading("patch", text="Patch")
        self.omit_tree.heading("region", text="Region")
        self.omit_tree.heading("hemisphere", text="Hemisphere")
        self.omit_tree.column("code", width=90, anchor="w")
        self.omit_tree.column("patch", width=70, anchor="w")
        self.omit_tree.column("region", width=170, anchor="w")
        self.omit_tree.column("hemisphere", width=90, anchor="w")
        omit_scroll = ttk.Scrollbar(omit_tab, orient="vertical", command=self.omit_tree.yview)
        self.omit_tree.configure(yscrollcommand=omit_scroll.set)
        self.omit_tree.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        omit_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 6), pady=6)

        omit_buttons = ttk.Frame(omit_tab)
        omit_buttons.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
        ttk.Button(omit_buttons, text="Remove selected", command=self._remove_selected_region).pack(side="left", padx=(0, 6))
        ttk.Button(omit_buttons, text="Clear current slice", command=self._clear_current_slice).pack(side="left", padx=(0, 6))
        ttk.Button(omit_buttons, text="Clear all", command=self._clear_all).pack(side="left")

        bottom = ttk.Frame(self)
        bottom.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        bottom.columnconfigure(0, weight=1)
        bottom.columnconfigure(1, weight=1)
        bottom.columnconfigure(2, weight=0)
        ttk.Label(bottom, textvariable=self.hover_var, anchor="w").grid(row=0, column=0, sticky="ew")
        ttk.Label(bottom, textvariable=self.status_var, anchor="e").grid(row=0, column=1, sticky="ew")
        ttk.Label(bottom, text=f"Version {APP_VERSION}", anchor="e", foreground="#5f6b7a").grid(
            row=0,
            column=2,
            sticky="e",
            padx=(8, 0),
        )

    def _browse_overlay_dir(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.overlay_dir_var.get() or Path.cwd())
        if selected:
            self.overlay_dir_var.set(selected)

    def _load_overlay_dir(self) -> None:
        overlay_dir = Path(self.overlay_dir_var.get().strip())
        if not overlay_dir:
            messagebox.showwarning("Missing overlay directory", "Please choose an overlay directory.")
            return
        try:
            dataset = load_overlay_dataset(overlay_dir)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        self.dataset = dataset
        session_path = session_path_for_overlay_dir(dataset.overlay_dir)
        if session_path.exists():
            try:
                self.session = normalize_session(dataset, load_session(session_path))
                self.status_var.set(f"Loaded session: {session_path}")
            except Exception as exc:
                LOG.exception("Failed to load omit session")
                self.session = create_default_session(dataset)
                self.status_var.set(f"Session load failed, using defaults: {exc}")
        else:
            self.session = create_default_session(dataset)
            self.status_var.set("Overlay loaded")
        self._populate_dataset_views()
        self._autosave_session()

    def _populate_dataset_views(self) -> None:
        assert self.dataset is not None
        assert self.session is not None
        self.slice_listbox.delete(0, tk.END)
        for slice_info in self.dataset.slices:
            self.slice_listbox.insert(tk.END, slice_info.display_name)

        self.channel_visibility.clear()
        self.channel_checkbuttons.clear()
        for child in self.channel_frame.winfo_children():
            child.destroy()
        for info in self.dataset.visible_channel_infos:
            var = tk.BooleanVar(value=info.content in self.session.visible_contents)
            self.channel_visibility[info.content] = var
            check = ttk.Checkbutton(
                self.channel_frame,
                text=self._channel_label(info),
                variable=var,
                command=self._on_channel_toggle,
            )
            check.pack(anchor="w", padx=4, pady=2)
            self.channel_checkbuttons.append(check)

        self._select_slice_by_key(self.session.selected_slice_key)

    def _channel_label(self, info: OverlayChannelInfo) -> str:
        if info.is_raw_image:
            return f"{info.source_channel or info.channel_name} raw image"
        if info.is_cell_roi:
            return f"{info.source_channel or info.channel_name} cell ROI"
        if info.is_overlap_roi:
            return "Matched overlap ROI"
        if info.is_outline:
            return "Brain region outline"
        if info.is_omit_mask:
            return "Imported omit mask"
        return info.content

    def _select_slice_by_key(self, slice_key: str) -> None:
        assert self.dataset is not None
        target_index = 0
        for index, slice_info in enumerate(self.dataset.slices):
            if slice_info.key == slice_key:
                target_index = index
                break
        self.slice_listbox.selection_clear(0, tk.END)
        self.slice_listbox.selection_set(target_index)
        self.slice_listbox.see(target_index)
        self._load_slice(self.dataset.slices[target_index])

    def _on_slice_select(self, _event) -> None:
        if self.dataset is None:
            return
        selection = self.slice_listbox.curselection()
        if not selection:
            return
        self._load_slice(self.dataset.slices[int(selection[0])])

    def _load_slice(self, slice_info: OverlaySliceInfo) -> None:
        assert self.dataset is not None
        assert self.session is not None
        if self.current_slice and self.current_slice.key == slice_info.key and self.current_stack is not None:
            self._refresh_region_list()
            self._render_current_slice()
            return
        self.current_slice = slice_info
        self.session.selected_slice_key = slice_info.key
        self.current_stack = load_overlay_stack(slice_info)
        self.slice_summary_var.set(
            f"Overlay: {slice_info.overlay_path.name}\nAnimal: {slice_info.animal_id}\nSection: {slice_info.section_id}"
        )
        self._refresh_region_list()
        self._render_current_slice()
        self._autosave_session()

    def _on_channel_toggle(self) -> None:
        if self.session is None:
            return
        self.session.visible_contents = [content for content, var in self.channel_visibility.items() if bool(var.get())]
        self._render_current_slice()
        self._autosave_session()

    def _render_current_slice(self) -> None:
        if self.dataset is None or self.session is None or self.current_slice is None or self.current_stack is None:
            self.canvas.delete("all")
            self.canvas.configure(scrollregion=(0, 0, 1, 1))
            return
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        image = render_qc_image(
            stack=self.current_stack,
            channels=self.dataset.channels,
            visible_contents=self.session.visible_contents,
            omitted_regions=self.session.omitted_regions_by_slice.get(self.current_slice.key, []),
        )
        transform = compute_canvas_transform(
            (image.height, image.width),
            canvas_width,
            canvas_height,
            zoom_factor=self._zoom_factor(),
        )
        resample = 0 if transform.scale >= 1.0 else 2
        rendered = image.resize((transform.draw_width, transform.draw_height), resample=resample)
        self._photo = ImageTk.PhotoImage(rendered)
        self.current_transform = transform
        self.canvas.delete("all")
        self.canvas.create_image(transform.offset_x, transform.offset_y, anchor="nw", image=self._photo)
        self.canvas.configure(scrollregion=(0, 0, transform.draw_width, transform.draw_height))
        if self._pending_center_image_xy is not None:
            self._center_view_on_image_xy(*self._pending_center_image_xy)
            self._pending_center_image_xy = None

    def _on_canvas_click(self, event) -> None:
        if self.dataset is None or self.session is None or self.current_slice is None or self.current_stack is None:
            return
        xy = self._canvas_to_image_xy(event.x, event.y)
        if xy is None:
            return
        selection = component_selection_at_xy(self.current_stack, self.dataset.channels, xy[0], xy[1])
        if selection is None:
            self.status_var.set("Clicked outside atlas region.")
            return
        omitted_regions = self.session.omitted_regions_by_slice.setdefault(self.current_slice.key, [])
        if selection in omitted_regions:
            omitted_regions.remove(selection)
            self.status_var.set(
                f"Removed region {selection.display_code} patch {selection.component_label} from omit list."
            )
        else:
            omitted_regions.append(selection)
            self.status_var.set(
                f"Added region {selection.display_code} patch {selection.component_label} to omit list."
            )
        omitted_regions.sort(key=lambda item: (int(item.display_code), int(item.component_label)))
        self._refresh_region_list()
        self._render_current_slice()
        self._autosave_session()

    def _on_canvas_hover(self, event) -> None:
        self._last_mouse_canvas_xy = (float(event.x), float(event.y))
        if self.dataset is None or self.current_stack is None:
            self.hover_var.set("Hover region: -")
            return
        xy = self._canvas_to_image_xy(event.x, event.y)
        if xy is None:
            self.hover_var.set("Hover region: -")
            return
        display_code = display_code_at_xy(self.current_stack, self.dataset.channels, xy[0], xy[1])
        if display_code == 0:
            self.hover_var.set("Hover region: outside atlas")
            return
        region = region_info_for_code(self.dataset.region_lookup, display_code)
        self.hover_var.set(
            f"Hover region: {region.region_name} | display_code={region.display_code} | hemisphere={region.hemisphere}"
        )

    def _canvas_to_image_xy(self, x_canvas: int, y_canvas: int) -> tuple[int, int] | None:
        if self.current_transform is None:
            return None
        actual_x = self.canvas.canvasx(x_canvas)
        actual_y = self.canvas.canvasy(y_canvas)
        return self.current_transform.to_image_xy(actual_x, actual_y)

    def _refresh_region_list(self) -> None:
        for item in self.omit_tree.get_children():
            self.omit_tree.delete(item)
        if self.dataset is None or self.session is None or self.current_slice is None:
            return
        selections = self.session.omitted_regions_by_slice.get(self.current_slice.key, [])
        for selection in selections:
            region = region_info_for_code(self.dataset.region_lookup, int(selection.display_code))
            self.omit_tree.insert(
                "",
                "end",
                values=(region.display_code, selection.component_label, region.region_name, region.hemisphere),
            )

    def _remove_selected_region(self) -> None:
        if self.dataset is None or self.session is None or self.current_slice is None:
            return
        selection_ids = self.omit_tree.selection()
        if not selection_ids:
            return
        selected_pairs = {
            (
                int(self.omit_tree.item(item_id, "values")[0]),
                int(self.omit_tree.item(item_id, "values")[1]),
            )
            for item_id in selection_ids
        }
        current_regions = self.session.omitted_regions_by_slice.setdefault(self.current_slice.key, [])
        self.session.omitted_regions_by_slice[self.current_slice.key] = [
            region
            for region in current_regions
            if (int(region.display_code), int(region.component_label)) not in selected_pairs
        ]
        self._refresh_region_list()
        self._render_current_slice()
        self._autosave_session()

    def _clear_current_slice(self) -> None:
        if self.session is None or self.current_slice is None:
            return
        self.session.omitted_regions_by_slice[self.current_slice.key] = []
        self._refresh_region_list()
        self._render_current_slice()
        self._autosave_session()

    def _clear_all(self) -> None:
        if self.session is None:
            return
        if not messagebox.askyesno("Clear all", "Clear omit selections for every slice?"):
            return
        for key in list(self.session.omitted_regions_by_slice):
            self.session.omitted_regions_by_slice[key] = []
        self._refresh_region_list()
        self._render_current_slice()
        self._autosave_session()

    def _load_session_dialog(self) -> None:
        selected = filedialog.askopenfilename(
            title="Load omit session",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            initialdir=self.overlay_dir_var.get() or Path.cwd(),
        )
        if not selected:
            return
        try:
            session = load_session(Path(selected))
            dataset = load_overlay_dataset(session.overlay_dir)
            self.overlay_dir_var.set(str(dataset.overlay_dir))
            self.dataset = dataset
            self.session = normalize_session(dataset, session)
            self._populate_dataset_views()
            self.status_var.set(f"Loaded session: {selected}")
        except Exception as exc:
            messagebox.showerror("Load session failed", str(exc))

    def _save_session_dialog(self) -> None:
        if self.dataset is None or self.session is None:
            return
        selected = filedialog.asksaveasfilename(
            title="Save omit session",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")],
            initialdir=str(session_path_for_overlay_dir(self.dataset.overlay_dir).parent),
            initialfile="omit_session.yaml",
        )
        if not selected:
            return
        save_session(self.session, Path(selected))
        self.status_var.set(f"Saved session: {selected}")

    def _autosave_session(self) -> None:
        if self.dataset is None or self.session is None:
            return
        path = session_path_for_overlay_dir(self.dataset.overlay_dir)
        save_session(self.session, path)

    def _finish(self) -> None:
        if self.dataset is None or self.session is None:
            messagebox.showwarning("Nothing to export", "Load an overlay directory first.")
            return
        output_dir = self.dataset.overlay_dir.parent / "omitByQDF3"
        try:
            outputs = export_omit_outputs(self.dataset, self.session, output_dir)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        row_count = len(build_omit_rows(self.dataset, self.session))
        self.status_var.set(f"Exported omit results: {outputs['csv']}")
        messagebox.showinfo(
            "QUINTdeepflow3 finished",
            f"Omit list exported to:\n{outputs['csv']}\n\nSelected patches: {row_count}\nImages: {outputs['images_dir']}",
        )

    def _zoom_factor(self) -> float | None:
        value = self.zoom_var.get().strip()
        if not value or value.lower() == "fit":
            return None
        if value.endswith("%"):
            try:
                return max(0.05, float(value[:-1]) / 100.0)
            except ValueError:
                return None
        try:
            return max(0.05, float(value))
        except ValueError:
            return None

    def _set_zoom_fit(self) -> None:
        self._pending_center_image_xy = self._anchor_image_xy()
        self.zoom_var.set("Fit")
        self._render_current_slice()

    def _step_zoom(self, direction: int) -> None:
        current = self.zoom_var.get().strip() or "Fit"
        if current.lower() == "fit":
            index = self.ZOOM_CHOICES.index("100%")
        else:
            try:
                index = self.ZOOM_CHOICES.index(current)
            except ValueError:
                index = self.ZOOM_CHOICES.index("100%")
        if current.lower() == "fit" and direction < 0:
            return
        new_index = max(1, min(len(self.ZOOM_CHOICES) - 1, index + int(direction)))
        self._pending_center_image_xy = self._anchor_image_xy()
        self.zoom_var.set(self.ZOOM_CHOICES[new_index])
        self._render_current_slice()

    def _on_zoom_combo_selected(self, _event) -> None:
        self._pending_center_image_xy = self._anchor_image_xy()
        self._render_current_slice()

    def _on_ctrl_mousewheel(self, event) -> str:
        self._last_mouse_canvas_xy = (float(event.x), float(event.y))
        self._step_zoom(1 if int(event.delta) > 0 else -1)
        return "break"

    def _anchor_image_xy(self) -> tuple[float, float] | None:
        if self.current_transform is None:
            return None
        if self._last_mouse_canvas_xy is not None:
            image_xy = self._canvas_to_image_xy(int(self._last_mouse_canvas_xy[0]), int(self._last_mouse_canvas_xy[1]))
            if image_xy is not None:
                return float(image_xy[0]), float(image_xy[1])
        center_canvas_x = self.canvas.winfo_width() / 2.0
        center_canvas_y = self.canvas.winfo_height() / 2.0
        image_xy = self._canvas_to_image_xy(int(center_canvas_x), int(center_canvas_y))
        if image_xy is None:
            return None
        return float(image_xy[0]), float(image_xy[1])

    def _center_view_on_image_xy(self, x_image: float, y_image: float) -> None:
        if self.current_transform is None:
            return
        total_width = max(1, int(self.current_transform.draw_width))
        total_height = max(1, int(self.current_transform.draw_height))
        viewport_width = max(1, int(self.canvas.winfo_width()))
        viewport_height = max(1, int(self.canvas.winfo_height()))
        target_x = float(x_image) * float(self.current_transform.scale)
        target_y = float(y_image) * float(self.current_transform.scale)
        if total_width > viewport_width:
            left = max(0.0, min(target_x - (viewport_width / 2.0), total_width - viewport_width))
            self.canvas.xview_moveto(left / max(total_width, 1))
        else:
            self.canvas.xview_moveto(0.0)
        if total_height > viewport_height:
            top = max(0.0, min(target_y - (viewport_height / 2.0), total_height - viewport_height))
            self.canvas.yview_moveto(top / max(total_height, 1))
        else:
            self.canvas.yview_moveto(0.0)


def launch_gui() -> None:
    """Launch the QUINTdeepflow3 GUI."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = QuintDeepflow3App()
    app.mainloop()
