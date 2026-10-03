"""
Interactive Patch Dataset Labeller for PILSS / YOLO Ballistics Verifier.
Provides a modern Tkinter GUI to:
1. Load image folder & resume labelling progress.
2. Interactively place bullet_hole and background patches with zoom/pan.
3. Track live class distribution & balance statistics.
4. Auto-crop patches and generate ready-to-train YOLO datasets (images, labels, data.yaml).
"""

import os
import sys
import json
import glob
import math
import shutil
import random
import tempfile
import logging
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

# Dual Logging: Console (for CLI/.exe) + Persistent Temp Log File
LOG_FILE = os.path.join(tempfile.gettempdir(), "cuidal_patch_labeller.log")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")
    ]
)
logger = logging.getLogger("CUIDAL")


class PatchAnnotationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CUIDAL — Custom Image Dataset Labeller (YOLO Patch Dataset Builder)")
        self.root.geometry("1400x900")
        self.root.minsize(1050, 680)

        # Open in maximized / windowed fullscreen on Windows
        try:
            self.root.state("zoomed")
        except Exception:
            pass

        # Style configuration
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # State Variables
        self.image_dir: str = ""
        self.export_dir: str = os.path.join(os.getcwd(), "yolo_patch_dataset")
        self.image_files: list[str] = []
        self.current_idx: int = 0
        
        # Manifest Data: { image_filename: [ { "cx": int, "cy": int, "size": int, "hole_diam": int, "class": str }, ... ] }
        self.manifest_file: str = ""
        self.manifest: dict = {}
        
        # Current Image & Canvas State
        self.cv_image: np.ndarray = None
        self.rgb_image: np.ndarray = None  # Cached RGB array (pre-converted once on load)
        self.tk_image: ImageTk.PhotoImage = None
        self.cached_zoom: float = -1.0
        self.img_item_id = None
        self.current_patches: list[dict] = []  # Patches for the current image
        self.selected_patch_idx: int = -1
        
        # Canvas transform
        self.zoom_level: float = 1.0
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.drag_start = None
        self.is_panning: bool = False
        self._initial_fit_done: bool = False

        # Progress and save state tracking
        self.has_unsaved_changes: bool = False
        self.dataset_exported: bool = False

        # Configuration variables
        self.active_class_var = tk.StringVar(value="bullet_hole")
        self.patch_size_var = tk.IntVar(value=64)  # 64x64 matching PILSS backend
        self.hole_diam_var = tk.IntVar(value=18)
        self.val_split_var = tk.IntVar(value=20)
        self.status_var = tk.StringVar(value="Ready. Open an image directory to start.")

        # Build UI layout
        self._build_ui()
        self._bind_events()

        # Intercept window close to protect unsaved progress
        self.root.protocol("WM_DELETE_WINDOW", self.on_close_request)

        # Restore last session after UI starts
        self.root.after(150, self.restore_last_session)

    def _build_ui(self):
        # Main Layout: Top Bar, Center Workspace (Canvas + Sidebar), Bottom Status Bar
        top_frame = ttk.Frame(self.root, padding=(8, 6))
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left Canvas Frame
        canvas_frame = ttk.Frame(self.main_paned, padding=2)
        self.main_paned.add(canvas_frame, weight=4)

        # Right Sidebar Container (Scrollable Frame)
        sidebar_container = ttk.Frame(self.main_paned, width=400)
        self.main_paned.add(sidebar_container, weight=1)

        # Bottom Status Bar
        bottom_frame = ttk.Frame(self.root, padding=(6, 3), relief=tk.SUNKEN)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(bottom_frame, textvariable=self.status_var, font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=6)

        # -------------------------------------------------------------
        # TOP BAR: Folder selectors and image progress
        # -------------------------------------------------------------
        ttk.Button(top_frame, text="📁 Open Images Directory", command=self.open_image_directory).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_frame, text="⚙️ Set Export Directory", command=self.set_export_directory).pack(side=tk.LEFT, padx=4)

        ttk.Separator(top_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        self.btn_prev = ttk.Button(top_frame, text="◀ Previous (A)", command=self.prev_image, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=4)

        self.progress_lbl = ttk.Label(top_frame, text="No Images Loaded", font=("Segoe UI", 10, "bold"))
        self.progress_lbl.pack(side=tk.LEFT, padx=12)

        self.btn_next = ttk.Button(top_frame, text="Next (D) ▶", command=self.next_image, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=4)

        ttk.Button(top_frame, text="💾 Save Patches (S)", command=self.save_current_image_annotations).pack(side=tk.LEFT, padx=8)

        # -------------------------------------------------------------
        # CANVAS WORKSPACE
        # -------------------------------------------------------------
        self.canvas = tk.Canvas(canvas_frame, bg="#1e1e24", cursor="crosshair", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # -------------------------------------------------------------
        # SCROLLABLE SIDEBAR WORKSPACE
        # -------------------------------------------------------------
        ttk_bg = self.style.lookup("TFrame", "background") or "#dcdad5"
        self.sb_canvas = tk.Canvas(sidebar_container, borderwidth=0, highlightthickness=0, bg=ttk_bg)
        self.sb_scrollbar = ttk.Scrollbar(sidebar_container, orient=tk.VERTICAL, command=self.sb_canvas.yview)
        self.sb_inner = ttk.Frame(self.sb_canvas, padding=(8, 6))

        self.sb_inner.bind(
            "<Configure>",
            lambda e: self.sb_canvas.configure(scrollregion=self.sb_canvas.bbox("all"))
        )

        self.sb_window_id = self.sb_canvas.create_window((0, 0), window=self.sb_inner, anchor="nw")
        self.sb_canvas.configure(yscrollcommand=self.sb_scrollbar.set)

        self.sb_canvas.bind("<Configure>", lambda e: self.sb_canvas.itemconfig(self.sb_window_id, width=e.width))

        self.sb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.sb_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Sidebar Mouse Wheel Routing
        self.sb_inner.bind("<Enter>", lambda e: self.root.bind_all("<MouseWheel>", self._on_sidebar_mousewheel))
        self.sb_inner.bind("<Leave>", lambda e: self.root.bind_all("<MouseWheel>", self.on_mouse_wheel))

        # -------------------------------------------------------------
        # SIDEBAR WIDGETS
        # -------------------------------------------------------------
        # 1. Active Patch Class Selector
        cls_box = ttk.LabelFrame(self.sb_inner, text=" 1. Patch Class Selector ", padding=(8, 6))
        cls_box.pack(fill=tk.X, pady=4)

        f_cls_btns = ttk.Frame(cls_box)
        f_cls_btns.pack(fill=tk.X, pady=2)
        r_hole = ttk.Radiobutton(f_cls_btns, text="🟢 Bullet Hole (Pos)", value="bullet_hole", variable=self.active_class_var)
        r_hole.pack(side=tk.LEFT, padx=4)

        r_bg = ttk.Radiobutton(f_cls_btns, text="🔴 Background (Neg)", value="background", variable=self.active_class_var)
        r_bg.pack(side=tk.LEFT, padx=4)

        ttk.Label(cls_box, text="Shortcut: Press '1' for Hole | '2' for Background", font=("Segoe UI", 8, "italic"), foreground="#777").pack(anchor=tk.W, pady=1)

        # 2. Patch Size & Geometry Settings
        geom_box = ttk.LabelFrame(self.sb_inner, text=" 2. Patch & Caliber Geometry ", padding=(8, 6))
        geom_box.pack(fill=tk.X, pady=4)

        f_presets = ttk.Frame(geom_box)
        f_presets.pack(fill=tk.X, pady=2)
        ttk.Label(f_presets, text="Presets:").pack(side=tk.LEFT, padx=2)
        ttk.Button(f_presets, text="64px (PILSS)", width=11, command=lambda: self._set_patch_size(64)).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_presets, text="96px", width=6, command=lambda: self._set_patch_size(96)).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_presets, text="128px", width=7, command=lambda: self._set_patch_size(128)).pack(side=tk.LEFT, padx=2)

        f_geom_inputs = ttk.Frame(geom_box)
        f_geom_inputs.pack(fill=tk.X, pady=2)

        ttk.Label(f_geom_inputs, text="Crop (px):").pack(side=tk.LEFT)
        ttk.Spinbox(f_geom_inputs, from_=32, to=512, increment=16, textvariable=self.patch_size_var, width=5, command=self._draw_patches).pack(side=tk.LEFT, padx=4)

        ttk.Label(f_geom_inputs, text="Hole (px):").pack(side=tk.LEFT, padx=(6, 0))
        ttk.Spinbox(f_geom_inputs, from_=6, to=120, increment=2, textvariable=self.hole_diam_var, width=5, command=self._draw_patches).pack(side=tk.LEFT, padx=4)

        # 3. Balance Statistics & Class Counts
        stat_box = ttk.LabelFrame(self.sb_inner, text=" 3. Live Balance & Dataset Counters ", padding=(8, 6))
        stat_box.pack(fill=tk.X, pady=4)

        self.lbl_curr_stats = ttk.Label(stat_box, text="Current Image: 0 Holes | 0 Background", font=("Segoe UI", 9, "bold"))
        self.lbl_curr_stats.pack(anchor=tk.W, pady=1)

        self.lbl_total_stats = ttk.Label(stat_box, text="Total Dataset: 0 Holes | 0 Background", font=("Segoe UI", 9, "bold"), foreground="#007acc")
        self.lbl_total_stats.pack(anchor=tk.W, pady=1)

        self.lbl_balance_ratio = ttk.Label(stat_box, text="Class Balance Ratio: N/A", font=("Segoe UI", 9))
        self.lbl_balance_ratio.pack(anchor=tk.W, pady=1)

        self.lbl_progress_summary = ttk.Label(stat_box, text="Images Completed: 0 / 0 (0%)", font=("Segoe UI", 9))
        self.lbl_progress_summary.pack(anchor=tk.W, pady=1)

        # 4. Patch Editing Actions
        edit_box = ttk.LabelFrame(self.sb_inner, text=" 4. Patch & View Actions ", padding=(8, 6))
        edit_box.pack(fill=tk.X, pady=4)

        btn_grid = ttk.Frame(edit_box)
        btn_grid.pack(fill=tk.X, pady=2)
        ttk.Button(btn_grid, text="🗑️ Delete Selected", command=self.delete_selected_patch).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btn_grid, text="🧹 Clear Image", command=self.clear_current_patches).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        f_zoom = ttk.Frame(edit_box)
        f_zoom.pack(fill=tk.X, pady=2)
        ttk.Button(f_zoom, text="🔍 Fit to Window", command=self.reset_zoom).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(f_zoom, text="1:1 Native Zoom", command=self.zoom_100).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        # 5. Image Directory File List (with Status)
        list_box = ttk.LabelFrame(self.sb_inner, text=" 5. Images in Folder ", padding=(8, 6))
        list_box.pack(fill=tk.X, pady=4)

        list_scroll = ttk.Scrollbar(list_box, orient=tk.VERTICAL)
        self.file_listbox = tk.Listbox(list_box, yscrollcommand=list_scroll.set, font=("Consolas", 9), selectmode=tk.SINGLE, height=7)
        list_scroll.config(command=self.file_listbox.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.file_listbox.bind("<<ListboxSelect>>", self.on_listbox_select)

        # 6. Generate Dataset & YOLO Crops
        exp_box = ttk.LabelFrame(self.sb_inner, text=" 6. YOLO Dataset Generation ", padding=(8, 6))
        exp_box.pack(fill=tk.X, pady=4)

        f_split = ttk.Frame(exp_box)
        f_split.pack(fill=tk.X, pady=2)
        ttk.Label(f_split, text="Val Split:").pack(side=tk.LEFT)
        ttk.Scale(f_split, from_=0, to=40, variable=self.val_split_var, orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        ttk.Label(f_split, textvariable=self.val_split_var, width=3).pack(side=tk.RIGHT)
        ttk.Label(f_split, text="%").pack(side=tk.RIGHT)

        self.btn_export = ttk.Button(
            exp_box,
            text="🚀 Generate YOLO Dataset (Crops + data.yaml)",
            command=self.export_yolo_dataset
        )
        self.btn_export.pack(fill=tk.X, pady=4)

    def _set_patch_size(self, sz: int):
        self.patch_size_var.set(sz)
        self._draw_patches()

    def _on_sidebar_mousewheel(self, event):
        self.sb_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_canvas_configure(self, event):
        if not self._initial_fit_done and self.cv_image is not None and event.width > 50 and event.height > 50:
            self._initial_fit_done = True
            self.reset_zoom()

    def _bind_events(self):
        # Canvas Mouse Interaction
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)  # Right click to delete patch
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)       # Zoom on Windows

        # Middle mouse button pan
        self.canvas.bind("<Button-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)
        self.canvas.bind("<ButtonRelease-2>", self.end_pan)

        # Keyboard shortcuts
        self.root.bind("<Key-1>", lambda e: self.active_class_var.set("bullet_hole"))
        self.root.bind("<Key-2>", lambda e: self.active_class_var.set("background"))
        self.root.bind("<Key-a>", lambda e: self.prev_image())
        self.root.bind("<Key-d>", lambda e: self.next_image())
        self.root.bind("<Key-s>", lambda e: self.save_current_image_annotations())
        self.root.bind("<Key-Delete>", lambda e: self.delete_selected_patch())

    # -------------------------------------------------------------------------
    # SESSION PERSISTENCE & CLOSE PROTECTION
    # -------------------------------------------------------------------------
    def get_app_config_file(self) -> str:
        import tempfile
        return os.path.join(tempfile.gettempdir(), "pilss_labeller_last_session.json")

    def save_app_state(self):
        if not self.image_dir:
            return
        cfg = {
            "last_image_dir": self.image_dir,
            "last_export_dir": self.export_dir,
            "last_idx": self.current_idx,
            "patch_size": self.patch_size_var.get(),
            "hole_diam": self.hole_diam_var.get(),
            "val_split": self.val_split_var.get()
        }
        try:
            with open(self.get_app_config_file(), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
        except Exception:
            pass

    def restore_last_session(self):
        cfg_file = self.get_app_config_file()
        if os.path.exists(cfg_file):
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                last_dir = cfg.get("last_image_dir", "")
                if last_dir and os.path.isdir(last_dir):
                    self.export_dir = cfg.get("last_export_dir", self.export_dir)
                    self.patch_size_var.set(cfg.get("patch_size", 64))
                    self.hole_diam_var.set(cfg.get("hole_diam", 18))
                    self.val_split_var.set(cfg.get("val_split", 20))
                    self.open_image_directory(folder=last_dir, resume_idx=cfg.get("last_idx", None))
                    self.status_var.set(f"Resumed previous session: {os.path.basename(last_dir)}")
            except Exception as e:
                print(f"Could not restore previous session: {e}")

    def on_close_request(self):
        if self.has_unsaved_changes:
            ans = messagebox.askyesnocancel(
                "Unsaved Progress",
                "You have unsaved patch annotations on the current image.\n\n"
                "Would you like to save your progress before exiting?\n\n"
                "• Click 'Yes' to Save Progress and Exit\n"
                "• Click 'No' to Discard Unsaved Changes and Exit\n"
                "• Click 'Cancel' to Keep Working",
                icon="warning"
            )
            if ans is True:  # Yes -> Save and Exit
                self.save_current_image_annotations(notify=False)
                self.save_app_state()
                self.root.destroy()
            elif ans is False:  # No -> Discard and Exit
                self.save_app_state()
                self.root.destroy()
            else:  # Cancel -> Do nothing, keep app open
                return
        else:
            # Everything is saved; save session state and exit cleanly
            self.save_app_state()
            self.root.destroy()

    # -------------------------------------------------------------------------
    # DIRECTORY & MANIFEST HANDLING
    # -------------------------------------------------------------------------
    def open_image_directory(self, folder: str = "", resume_idx: int = None):
        if not folder:
            folder = filedialog.askdirectory(title="Select Folder Containing Target Images")
        if not folder or not os.path.isdir(folder):
            return

        self.image_dir = folder
        extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff")
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(self.image_dir, ext)))
            files.extend(glob.glob(os.path.join(self.image_dir, ext.upper())))

        files = sorted(list(set(files)))
        if not files:
            messagebox.showwarning("No Images Found", f"No image files found in:\n{folder}")
            return

        self.image_files = files
        self.manifest_file = os.path.join(self.image_dir, "patch_labeller_manifest.json")
        self.load_manifest()

        # Update UI listbox
        self._refresh_file_listbox()

        # Determine target image index to open
        target_idx = 0
        if resume_idx is not None and 0 <= resume_idx < len(self.image_files):
            target_idx = resume_idx
        elif "_last_active_index" in self.manifest and isinstance(self.manifest["_last_active_index"], int) and 0 <= self.manifest["_last_active_index"] < len(self.image_files):
            target_idx = self.manifest["_last_active_index"]
        else:
            # Find first unlabelled image
            for i, f in enumerate(self.image_files):
                bname = os.path.basename(f)
                patches = self.manifest.get(bname, [])
                if not isinstance(patches, list) or len(patches) == 0:
                    target_idx = i
                    break

        self.load_image(target_idx)
        self.btn_prev.config(state=tk.NORMAL)
        self.btn_next.config(state=tk.NORMAL)
        self.update_stats()
        self.save_app_state()

    def set_export_directory(self):
        folder = filedialog.askdirectory(title="Select Export Directory for YOLO Dataset", initialdir=self.export_dir)
        if folder:
            self.export_dir = folder
            self.status_var.set(f"Export directory set to: {self.export_dir}")
            self.save_app_state()

    def load_manifest(self):
        self.manifest = {}
        if os.path.exists(self.manifest_file):
            try:
                with open(self.manifest_file, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
                n_ann = len([k for k in self.manifest if not k.startswith("_")])
                self.status_var.set(f"Loaded existing progress manifest: {n_ann} images recorded.")
            except Exception as e:
                self.status_var.set(f"Could not load manifest: {e}")

    def save_manifest(self):
        if not self.manifest_file:
            return
        try:
            self.manifest["_last_active_index"] = self.current_idx
            with open(self.manifest_file, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=2)
        except Exception as e:
            print(f"Error saving manifest: {e}")

    def _refresh_file_listbox(self):
        self.file_listbox.delete(0, tk.END)
        for i, f in enumerate(self.image_files):
            bname = os.path.basename(f)
            patches = self.manifest.get(bname, [])
            n_patches = len(patches) if isinstance(patches, list) else 0
            status = f"✓ [{n_patches:02d} patches]" if n_patches > 0 else "○ [unlabelled]"
            self.file_listbox.insert(tk.END, f"{status} {bname}")

    def on_listbox_select(self, event):
        sel = self.file_listbox.curselection()
        if sel:
            idx = sel[0]
            if idx != self.current_idx:
                self.save_current_image_annotations(notify=False)
                self.load_image(idx)

    # -------------------------------------------------------------------------
    # IMAGE NAVIGATION & LOADING
    # -------------------------------------------------------------------------
    def load_image(self, index: int):
        if not (0 <= index < len(self.image_files)):
            return

        self.current_idx = index
        img_path = self.image_files[index]
        bname = os.path.basename(img_path)

        # Load OpenCV image
        self.cv_image = cv2.imread(img_path)
        if self.cv_image is None:
            messagebox.showerror("Image Load Error", f"Could not open image: {img_path}")
            return

        # Pre-convert to RGB once per image load
        self.rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        self.cached_zoom = -1.0
        self.tk_image = None
        self.img_item_id = None
        self.canvas.delete("all")

        # Load existing patches for this image
        raw_patches = self.manifest.get(bname, [])
        if isinstance(raw_patches, list):
            self.current_patches = [dict(p) for p in raw_patches if isinstance(p, dict)]
        else:
            self.current_patches = []
        self.selected_patch_idx = -1
        self.has_unsaved_changes = False

        # Reset zoom / pan to fit
        self.reset_zoom()

        # Update progress header and listbox selection
        self.progress_lbl.config(text=f"Image {index + 1} / {len(self.image_files)}: {bname}")
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(index)
        self.file_listbox.see(index)

        self.update_stats()
        self.redraw_canvas(zoom_changed=True)

    def prev_image(self):
        if self.current_idx > 0:
            self.save_current_image_annotations(notify=False)
            self.load_image(self.current_idx - 1)

    def next_image(self):
        if self.current_idx < len(self.image_files) - 1:
            self.save_current_image_annotations(notify=False)
            self.load_image(self.current_idx + 1)
        else:
            self.save_current_image_annotations(notify=True)

    def save_current_image_annotations(self, notify: bool = True):
        if not self.image_files or self.current_idx >= len(self.image_files):
            return

        bname = os.path.basename(self.image_files[self.current_idx])
        if self.current_patches:
            self.manifest[bname] = [dict(p) for p in self.current_patches]
        else:
            if bname in self.manifest:
                del self.manifest[bname]

        self.has_unsaved_changes = False
        self.save_manifest()
        self.save_app_state()
        self._refresh_file_listbox()
        self.file_listbox.selection_set(self.current_idx)
        self.update_stats()
        if notify:
            self.status_var.set(f"Saved {len(self.current_patches)} patches for '{bname}'.")

    # -------------------------------------------------------------------------
    # HIGH-PERFORMANCE CANVAS RENDERING, ZOOM & PAN
    # -------------------------------------------------------------------------
    def reset_zoom(self):
        if self.cv_image is None:
            return
        h, w = self.cv_image.shape[:2]
        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())
        scale_w = cw / float(w)
        scale_h = ch / float(h)
        self.zoom_level = min(scale_w, scale_h) * 0.95
        self.pan_x = (cw - w * self.zoom_level) / 2.0
        self.pan_y = (ch - h * self.zoom_level) / 2.0
        self.redraw_canvas(zoom_changed=True)

    def zoom_100(self):
        self.zoom_level = 1.0
        self.pan_x = 20.0
        self.pan_y = 20.0
        self.redraw_canvas(zoom_changed=True)

    def _update_image_display(self, zoom_changed=False):
        if self.rgb_image is None:
            return

        h, w = self.rgb_image.shape[:2]

        # Only re-resize when zoom changes (fast C++ OpenCV SIMD resize)
        if zoom_changed or abs(self.zoom_level - self.cached_zoom) > 1e-4 or self.tk_image is None:
            new_w = max(1, int(round(w * self.zoom_level)))
            new_h = max(1, int(round(h * self.zoom_level)))

            interp = cv2.INTER_LINEAR if self.zoom_level < 1.0 else cv2.INTER_NEAREST
            resized_rgb = cv2.resize(self.rgb_image, (new_w, new_h), interpolation=interp)
            self.tk_image = ImageTk.PhotoImage(Image.fromarray(resized_rgb))
            self.cached_zoom = self.zoom_level

        # Reposition image item on canvas
        if self.img_item_id is not None and self.canvas.find_withtag(self.img_item_id):
            self.canvas.coords(self.img_item_id, self.pan_x, self.pan_y)
            if zoom_changed:
                self.canvas.itemconfig(self.img_item_id, image=self.tk_image)
        else:
            self.img_item_id = self.canvas.create_image(
                self.pan_x, self.pan_y, anchor=tk.NW, image=self.tk_image, tags="bg_image"
            )
            self.canvas.tag_lower(self.img_item_id)

    def _draw_patches(self):
        # Ultra-fast: Only redraws vector shapes with tag 'patch_tag' (< 0.05ms)
        self.canvas.delete("patch_tag")
        if self.cv_image is None:
            return

        for i, patch in enumerate(self.current_patches):
            cx = patch["cx"]
            cy = patch["cy"]
            psz = patch.get("size", self.patch_size_var.get())
            hdiam = patch.get("hole_diam", self.hole_diam_var.get())
            pcls = patch.get("class", "bullet_hole")

            # Image space to Canvas screen space
            sx = self.pan_x + cx * self.zoom_level
            sy = self.pan_y + cy * self.zoom_level
            s_half = (psz / 2.0) * self.zoom_level
            s_hr = (hdiam / 2.0) * self.zoom_level

            x1, y1 = sx - s_half, sy - s_half
            x2, y2 = sx + s_half, sy + s_half

            # Color styling
            is_selected = (i == self.selected_patch_idx)
            if pcls == "bullet_hole":
                box_color = "#50fa7b" if not is_selected else "#ffff00"  # Bright Green / Yellow
                label_txt = f"Hole #{i+1}"
            else:
                box_color = "#ff5555" if not is_selected else "#ffff00"  # Bright Red / Yellow
                label_txt = f"Bg #{i+1}"

            # Draw outer patch crop bounding box
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=box_color,
                width=3 if is_selected else 2,
                dash=(4, 2) if pcls == "background" else None,
                tags="patch_tag"
            )

            # Draw center point and caliber circle for bullet holes
            if pcls == "bullet_hole":
                self.canvas.create_oval(
                    sx - s_hr, sy - s_hr, sx + s_hr, sy + s_hr,
                    outline="#ff79c6", width=2, tags="patch_tag"
                )
                self.canvas.create_oval(
                    sx - 3, sy - 3, sx + 3, sy + 3,
                    fill="#50fa7b", outline="black", tags="patch_tag"
                )
            else:
                self.canvas.create_line(
                    sx - 4, sy - 4, sx + 4, sy + 4,
                    fill="#ff5555", width=2, tags="patch_tag"
                )
                self.canvas.create_line(
                    sx - 4, sy + 4, sx + 4, sy - 4,
                    fill="#ff5555", width=2, tags="patch_tag"
                )

            # Draw label tag
            self.canvas.create_text(
                x1 + 4, y1 - 10,
                text=label_txt,
                anchor=tk.NW,
                fill=box_color,
                font=("Segoe UI", 9, "bold"),
                tags="patch_tag"
            )

    def redraw_canvas(self, zoom_changed=False):
        self._update_image_display(zoom_changed=zoom_changed)
        self._draw_patches()

    # -------------------------------------------------------------------------
    # CANVAS INTERACTIONS: PLACING & SELECTING PATCHES
    # -------------------------------------------------------------------------
    def on_canvas_click(self, event):
        if self.cv_image is None:
            return

        # Check if clicked on an existing patch
        clicked_idx = self._find_patch_at_screen_pos(event.x, event.y)
        if clicked_idx != -1:
            self.selected_patch_idx = clicked_idx
            self.drag_start = (event.x, event.y)
            self._draw_patches()
            return

        # Otherwise: Place a new patch at clicked location
        img_x = (event.x - self.pan_x) / self.zoom_level
        img_y = (event.y - self.pan_y) / self.zoom_level
        h, w = self.cv_image.shape[:2]

        if 0 <= img_x < w and 0 <= img_y < h:
            new_patch = {
                "cx": int(round(img_x)),
                "cy": int(round(img_y)),
                "size": self.patch_size_var.get(),
                "hole_diam": self.hole_diam_var.get(),
                "class": self.active_class_var.get()
            }
            self.current_patches.append(new_patch)
            self.selected_patch_idx = len(self.current_patches) - 1
            self.has_unsaved_changes = True
            self.update_stats()
            self._draw_patches()
            self.status_var.set(f"Added {new_patch['class']} patch at ({new_patch['cx']}, {new_patch['cy']}).")

    def on_canvas_drag(self, event):
        if self.selected_patch_idx != -1 and self.drag_start is not None:
            # Move selected patch
            dx = (event.x - self.drag_start[0]) / self.zoom_level
            dy = (event.y - self.drag_start[1]) / self.zoom_level
            patch = self.current_patches[self.selected_patch_idx]
            patch["cx"] = int(round(patch["cx"] + dx))
            patch["cy"] = int(round(patch["cy"] + dy))
            self.drag_start = (event.x, event.y)
            self.has_unsaved_changes = True
            # Ultra-fast: only updates vector coordinates without touching image
            self._draw_patches()

    def on_canvas_release(self, event):
        self.drag_start = None

    def on_canvas_right_click(self, event):
        # Delete patch under right click
        idx = self._find_patch_at_screen_pos(event.x, event.y)
        if idx != -1:
            del_patch = self.current_patches.pop(idx)
            self.selected_patch_idx = -1
            self.has_unsaved_changes = True
            self.update_stats()
            self._draw_patches()
            self.status_var.set(f"Deleted {del_patch['class']} patch.")

    def _find_patch_at_screen_pos(self, sx: float, sy: float) -> int:
        for i in reversed(range(len(self.current_patches))):
            patch = self.current_patches[i]
            px = self.pan_x + patch["cx"] * self.zoom_level
            py = self.pan_y + patch["cy"] * self.zoom_level
            half = (patch.get("size", 128) / 2.0) * self.zoom_level
            if (px - half <= sx <= px + half) and (py - half <= sy <= py + half):
                return i
        return -1

    def delete_selected_patch(self):
        if 0 <= self.selected_patch_idx < len(self.current_patches):
            self.current_patches.pop(self.selected_patch_idx)
            self.selected_patch_idx = -1
            self.has_unsaved_changes = True
            self.update_stats()
            self._draw_patches()

    def clear_current_patches(self):
        if self.current_patches:
            if messagebox.askyesno("Clear Patches", "Remove all patches from this image?"):
                self.current_patches.clear()
                self.selected_patch_idx = -1
                self.has_unsaved_changes = True
                self.update_stats()
                self._draw_patches()

    def on_mouse_wheel(self, event):
        # Zoom centered on mouse pointer
        if self.cv_image is None:
            return
        factor = 1.15 if event.delta > 0 else 0.85
        new_zoom = float(np.clip(self.zoom_level * factor, 0.05, 10.0))

        # Adjust pan to keep point under cursor invariant
        self.pan_x = event.x - (event.x - self.pan_x) * (new_zoom / self.zoom_level)
        self.pan_y = event.y - (event.y - self.pan_y) * (new_zoom / self.zoom_level)
        self.zoom_level = new_zoom
        self.redraw_canvas(zoom_changed=True)

    def start_pan(self, event):
        self.is_panning = True
        self.pan_start = (event.x - self.pan_x, event.y - self.pan_y)

    def do_pan(self, event):
        if self.is_panning:
            self.pan_x = event.x - self.pan_start[0]
            self.pan_y = event.y - self.pan_start[1]
            # Ultra-fast pan: Reposition existing image item and redraw patches
            if self.img_item_id is not None and self.canvas.find_withtag(self.img_item_id):
                self.canvas.coords(self.img_item_id, self.pan_x, self.pan_y)
            self._draw_patches()

    def end_pan(self, event):
        self.is_panning = False

    # -------------------------------------------------------------------------
    # STATS & BALANCE COUNTER
    # -------------------------------------------------------------------------
    def update_stats(self):
        # Current image counts
        curr_holes = sum(1 for p in self.current_patches if p.get("class") == "bullet_hole")
        curr_bg = sum(1 for p in self.current_patches if p.get("class") == "background")
        self.lbl_curr_stats.config(text=f"Current Image: {curr_holes} Holes | {curr_bg} Background")

        # Total dataset counts across all images in manifest
        total_holes = 0
        total_bg = 0
        labelled_imgs = 0

        # Include unsaved current patches in live total
        active_bname = os.path.basename(self.image_files[self.current_idx]) if self.image_files else ""
        temp_manifest = dict(self.manifest)
        if active_bname:
            temp_manifest[active_bname] = self.current_patches

        for bname, patches in temp_manifest.items():
            if bname.startswith("_") or not isinstance(patches, list):
                continue
            if patches:
                labelled_imgs += 1
                for p in patches:
                    if isinstance(p, dict):
                        if p.get("class") == "bullet_hole":
                            total_holes += 1
                        else:
                            total_bg += 1

        self.lbl_total_stats.config(text=f"Total Dataset: {total_holes} Holes | {total_bg} Background")

        if total_holes > 0 and total_bg > 0:
            ratio = total_holes / float(total_bg)
            balance_desc = " (Perfect 1:1)" if 0.9 <= ratio <= 1.1 else (" (Slight Imbalance)" if 0.7 <= ratio <= 1.4 else " (Imbalanced!)")
            self.lbl_balance_ratio.config(text=f"Class Ratio: {ratio:.2f} : 1{balance_desc}")
        else:
            self.lbl_balance_ratio.config(text="Class Ratio: Waiting for both classes")

        n_total = len(self.image_files)
        pct = (labelled_imgs / float(n_total) * 100.0) if n_total > 0 else 0.0
        self.lbl_progress_summary.config(text=f"Images Completed: {labelled_imgs} / {n_total} ({pct:.1f}%)")

    # -------------------------------------------------------------------------
    # YOLO DATASET EXPORT ENGINE
    # -------------------------------------------------------------------------
    def export_yolo_dataset(self):
        self.save_current_image_annotations(notify=False)

        if not self.manifest:
            messagebox.showwarning("No Patches", "No patches have been defined yet! Add patches before generating dataset.")
            return

        total_patches = sum(len(patches) for k, patches in self.manifest.items() if not k.startswith("_") and isinstance(patches, list))
        if total_patches == 0:
            messagebox.showwarning("No Patches", "All images currently have 0 patches.")
            return

        val_pct = self.val_split_var.get() / 100.0
        export_path = Path(self.export_dir).resolve()

        # Confirm with user
        msg = (
            f"Ready to generate YOLO patch dataset:\n\n"
            f"• Total Patches: {total_patches}\n"
            f"• Target Directory: {export_path}\n"
            f"• Train / Val Split: {int((1.0 - val_pct)*100)}% Train / {int(val_pct*100)}% Val\n\n"
            f"Proceed with cropping and formatting?"
        )
        if not messagebox.askyesno("Export Dataset", msg):
            return

        # Prepare directories
        train_img_dir = export_path / "images" / "train"
        val_img_dir = export_path / "images" / "val"
        train_lbl_dir = export_path / "labels" / "train"
        val_lbl_dir = export_path / "labels" / "val"

        for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Collect all patch tasks
        tasks = []
        for bname, patches in self.manifest.items():
            if bname.startswith("_") or not isinstance(patches, list):
                continue
            img_path = os.path.join(self.image_dir, bname)
            if not os.path.exists(img_path):
                continue
            for p_idx, p in enumerate(patches):
                if isinstance(p, dict):
                    tasks.append({
                        "img_path": img_path,
                        "bname": bname,
                        "patch_idx": p_idx,
                        "patch_data": p
                    })

        # Shuffle and split into Train / Val
        random.seed(42)
        random.shuffle(tasks)
        n_val = int(len(tasks) * val_pct)
        val_tasks = tasks[:n_val]
        train_tasks = tasks[n_val:]

        def process_tasks(task_list, img_dst_dir, lbl_dst_dir):
            for t in task_list:
                img = cv2.imread(t["img_path"])
                if img is None:
                    continue
                h, w = img.shape[:2]
                p = t["patch_data"]
                cx = p["cx"]
                cy = p["cy"]
                psz = p.get("size", 128)
                hdiam = p.get("hole_diam", 24)
                pcls = p.get("class", "bullet_hole")

                # Crop square patch
                half = psz // 2
                x1 = max(0, cx - half)
                y1 = max(0, cy - half)
                x2 = min(w, cx + half)
                y2 = min(h, cy + half)

                patch = img[y1:y2, x1:x2]

                # If on boundary, pad with replication to ensure exact size
                ph, pw = patch.shape[:2]
                if ph != psz or pw != psz:
                    p_top = max(0, half - cy)
                    p_bot = max(0, (cy + half) - h)
                    p_left = max(0, half - cx)
                    p_right = max(0, (cx + half) - w)
                    patch = cv2.copyMakeBorder(patch, p_top, p_bot, p_left, p_right, cv2.BORDER_REPLICATE)
                    patch = cv2.resize(patch, (psz, psz))

                # Unique patch filename
                stem = Path(t["bname"]).stem
                patch_name = f"{stem}_p{t['patch_idx']:03d}_{pcls}"
                img_out_file = img_dst_dir / f"{patch_name}.png"
                lbl_out_file = lbl_dst_dir / f"{patch_name}.txt"

                cv2.imwrite(str(img_out_file), patch)

                # Generate YOLO Annotation: Class 0 = bullet_hole
                # In normalized patch coordinates (0.0 to 1.0)
                if pcls == "bullet_hole":
                    # Bullet hole is centered in the crop: (0.5, 0.5)
                    norm_cx = 0.5
                    norm_cy = 0.5
                    norm_w = float(np.clip(hdiam / float(psz), 0.05, 0.95))
                    norm_h = float(np.clip(hdiam / float(psz), 0.05, 0.95))
                    with open(lbl_out_file, "w", encoding="utf-8") as lf:
                        lf.write(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                else:
                    # Empty text file for negative / background sample
                    with open(lbl_out_file, "w", encoding="utf-8") as lf:
                        pass  # Empty file instructs YOLO this is pure background

        # Execute extraction
        process_tasks(train_tasks, train_img_dir, train_lbl_dir)
        process_tasks(val_tasks, val_img_dir, val_lbl_dir)

        # Generate data.yaml
        yaml_content = f"""# PILSS YOLO Bullet Verifier Dataset Configuration
path: {export_path.as_posix()}
train: images/train
val: images/val

names:
  0: bullet_hole
"""
        yaml_file = export_path / "data.yaml"
        with open(yaml_file, "w", encoding="utf-8") as yf:
            yf.write(yaml_content)

        # Generate ready-to-run train script template
        train_script = f"""# Auto-generated YOLO training script for PILSS Verifier
from ultralytics import YOLO

def main():
    model = YOLO("yolov8s.pt")
    model.train(
        data=r"{yaml_file.as_posix()}",
        epochs=100,
        imgsz={self.patch_size_var.get()},
        batch=32,
        mosaic=0.0,      # Disabled to maintain circular hole geometry
        degrees=180.0,   # Full rotation invariance
        save=True,
        project="pilss_yolo_runs",
        name="bullet_verifier_yolov8s"
    )

if __name__ == "__main__":
    main()
"""
        train_script_file = export_path / "train_yolo.py"
        with open(train_script_file, "w", encoding="utf-8") as sf:
            sf.write(train_script)

        self.has_unsaved_changes = False
        self.dataset_exported = True
        self.status_var.set(f"Successfully generated YOLO dataset at: {export_path}")
        messagebox.showinfo(
            "Dataset Generated!",
            f"YOLO Patch Dataset generated successfully!\n\n"
            f"• Train Samples: {len(train_tasks)}\n"
            f"• Val Samples: {len(val_tasks)}\n"
            f"• Output Path: {export_path}\n"
            f"• Config File: {yaml_file}\n"
            f"• Training Script: {train_script_file}"
        )


def main():
    print("=" * 65)
    print("  PILSS / CUIDAL Custom Patch Dataset Labeller v1.0.0")
    print("  GitHub: https://github.com/Joel-GJA/CUIDAL")
    print(f"  Live Log File: {LOG_FILE}")
    print("=" * 65)
    logger.info("Starting PILSS Patch Labeller application...")
    root = tk.Tk()
    app = PatchAnnotationApp(root)
    root.mainloop()
    logger.info("Application closed cleanly.")


if __name__ == "__main__":
    main()
