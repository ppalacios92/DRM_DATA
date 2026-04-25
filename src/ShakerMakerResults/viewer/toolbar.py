"""Top toolbar for the interactive viewer — view presets, capture and overlays."""

from __future__ import annotations

from ._imports import require_viewer_dependencies

_, _, _, QtCore, _, QtWidgets = require_viewer_dependencies()

try:
    import qtawesome as qta
    _HAS_QTA = True
except ImportError:
    _HAS_QTA = False

# ── ffmpeg path ───────────────────────────────────────────────────────────────
# Set this to your local ffmpeg.exe if ffmpeg is not on the system PATH.
# Leave as None to rely on the system PATH or imageio-ffmpeg auto-detection.
FFMPEG_EXE: str | None = r"C:\Dropbox\01. Brain\10. Ph.D U ANDES\21. ffmpeg\ffmpeg-2025-09-18-git-c373636f55-essentials_build\bin\ffmpeg.exe"
# ─────────────────────────────────────────────────────────────────────────────


def _icon(name: str, color: str = "#404040"):
    if _HAS_QTA:
        return qta.icon(name, color=color)
    return None


# (label, tooltip, azimuth_offset)
_ISO_PRESETS = [
    ("NE", "Isometric — North-East",  0),
    ("NW", "Isometric — North-West", 90),
    ("SW", "Isometric — South-West", 180),
    ("SE", "Isometric — South-East", 270),
]

# (label, tooltip, icon_name, plotter_method, extra_kwargs)
_VIEW_PRESETS = [
    ("Top",   "Top view (+Z)",    "mdi.arrow-collapse-up",   "view_xy",  {}),
    ("Bot",   "Bottom view (−Z)", "mdi.arrow-collapse-down", "view_xy",  {"negative": True}),
    ("Front", "Front view (−Y)",  "mdi.arrow-expand-up",     "view_xz",  {}),
    ("Back",  "Back view (+Y)",   "mdi.arrow-expand-down",   "view_xz",  {"negative": True}),
    ("Left",  "Left view (−X)",   "mdi.arrow-expand-left",   "view_yz",  {"negative": True}),
    ("Right", "Right view (+X)",  "mdi.arrow-expand-right",  "view_yz",  {}),
]


class ViewerToolBar(QtWidgets.QWidget):
    """Horizontal icon toolbar placed below the header bar.

    Parameters
    ----------
    multi_view:
        The :class:`~.multi_view.MultiViewArea` that owns all view panes.
        View-preset and capture operations always target its *active* pane,
        so the toolbar naturally follows whichever pane the user last clicked.
    """

    def __init__(self, multi_view, parent=None):
        super().__init__(parent)
        self._multi_view = multi_view
        self._recording = False
        self._recording_plotter = None   # captured at open_movie() call time
        self._show_bbox = False
        self._show_axes = True

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        layout.setSpacing(1)

        # ── ISO views (4 angles) ─────────────────────────────────────────────
        self._add_sep(layout)
        for label, tip, azimuth in _ISO_PRESETS:
            btn = self._btn(label, "mdi.rotate-3d-variant", tip)
            btn.clicked.connect(self._iso_view_cb(azimuth))
            layout.addWidget(btn)

        # ── Orthogonal views ─────────────────────────────────────────────────
        self._add_sep(layout)
        for label, tip, icon_name, method, kwargs in _VIEW_PRESETS:
            btn = self._btn(label, icon_name, tip)
            btn.clicked.connect(self._view_cb(method, kwargs))
            layout.addWidget(btn)

        # ── Camera ───────────────────────────────────────────────────────────
        self._add_sep(layout)

        fit_btn = self._btn(
            "Fit", "mdi.fit-to-page-outline", "Fit all — reset camera to show full model"
        )
        fit_btn.clicked.connect(self._fit_all)
        layout.addWidget(fit_btn)

        self._ortho_btn = self._btn(
            "Ortho", "mdi.grid",
            "Toggle orthographic / perspective projection",
            checkable=True,
        )
        self._ortho_btn.toggled.connect(self._toggle_ortho)
        layout.addWidget(self._ortho_btn)

        # ── Capture ───────────────────────────────────────────────────────────
        self._add_sep(layout)

        shot_btn = self._btn("", "mdi.camera-outline", "Save screenshot as PNG")
        shot_btn.clicked.connect(self._screenshot)
        layout.addWidget(shot_btn)

        self._record_btn = self._btn(
            "", "mdi.record-circle-outline",
            "Record animation to MP4 / GIF",
            checkable=True,
        )
        self._record_btn.toggled.connect(self._toggle_record)
        layout.addWidget(self._record_btn)

        # ── Overlays ─────────────────────────────────────────────────────────
        self._add_sep(layout)

        self._axes_btn = self._btn(
            "Axes", "mdi.axis-arrow",
            "Toggle orientation axes widget",
            checkable=True,
        )
        self._axes_btn.setChecked(True)
        self._axes_btn.toggled.connect(self._toggle_axes)
        layout.addWidget(self._axes_btn)

        self._bbox_btn = self._btn(
            "BBox", "mdi.crop-square",
            "Toggle bounding box",
            checkable=True,
        )
        self._bbox_btn.toggled.connect(self._toggle_bbox)
        layout.addWidget(self._bbox_btn)

        layout.addStretch(1)

        self.setStyleSheet("ViewerToolBar { border-bottom: 1px solid #d0d0d0; }")

    # ── Active-plotter access ─────────────────────────────────────────────────

    @property
    def _plotter(self):
        """Active pane's plotter — re-evaluated on every access."""
        return self._multi_view.active_plotter

    # ── View presets ──────────────────────────────────────────────────────────

    def _view_cb(self, method_name: str, kwargs: dict):
        """Return a callback that calls *method_name* on the active plotter."""
        def _cb():
            p = self._plotter
            if p is None:
                return
            fn = getattr(p, method_name, None)
            if fn is not None:
                try:
                    fn(**kwargs)
                    p.render()
                except Exception:
                    pass
        return _cb

    def _iso_view_cb(self, azimuth_extra: float):
        """Return a callback for an isometric view rotated by *azimuth_extra*°."""
        def _cb():
            p = self._plotter
            if p is None:
                return
            try:
                p.view_isometric()
                if azimuth_extra:
                    p.camera.Azimuth(azimuth_extra)
                    p.reset_camera_clipping_range()
                p.render()
            except Exception:
                pass
        return _cb

    def _fit_all(self):
        p = self._plotter
        if p is None:
            return
        try:
            p.reset_camera()
            p.render()
        except Exception:
            pass

    # ── Camera ────────────────────────────────────────────────────────────────

    def _toggle_ortho(self, checked: bool):
        p = self._plotter
        if p is None:
            return
        try:
            if checked:
                p.enable_parallel_projection()
            else:
                p.disable_parallel_projection()
            p.render()
        except Exception:
            pass

    # ── Capture ───────────────────────────────────────────────────────────────

    def _screenshot(self):
        p = self._plotter
        if p is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Screenshot", "screenshot.png",
            "PNG image (*.png);;All files (*)",
        )
        if not path:
            return
        try:
            p.screenshot(path, transparent_background=False)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Screenshot failed", str(exc))

    def _toggle_record(self, checked: bool):
        if checked:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        p = self._plotter
        if p is None:
            self._uncheck_record()
            return

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Recording", "animation.mp4",
            "MP4 video (*.mp4);;GIF animation (*.gif);;All files (*)",
        )
        if not path:
            self._uncheck_record()
            return

        try:
            if FFMPEG_EXE:
                import os
                os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_EXE
            p.open_movie(path, framerate=12, quality=5)
            # Capture this specific plotter — write/stop must use the same one.
            self._recording_plotter = p
            self._recording = True
            if _HAS_QTA:
                self._record_btn.setIcon(
                    qta.icon("mdi.stop-circle-outline", color="#cc2020")
                )
            self._record_btn.setToolTip("Stop recording  ● REC")
        except Exception as exc:
            self._uncheck_record()
            QtWidgets.QMessageBox.warning(
                self, "Recording failed",
                f"{exc}\n\nffmpeg may not be installed. "
                "Run:  pip install imageio-ffmpeg",
            )

    def _stop_recording(self):
        if self._recording and self._recording_plotter is not None:
            try:
                self._recording_plotter.close_movie()
            except Exception:
                pass
        self._recording = False
        self._recording_plotter = None
        if _HAS_QTA:
            self._record_btn.setIcon(
                qta.icon("mdi.record-circle-outline", color="#404040")
            )
        self._record_btn.setToolTip("Record animation to MP4 / GIF")

    def _uncheck_record(self):
        """Silently uncheck the record button (user cancelled or error)."""
        self._record_btn.blockSignals(True)
        self._record_btn.setChecked(False)
        self._record_btn.blockSignals(False)

    def write_frame_if_recording(self):
        """Write the current render as one video frame after each playback step."""
        if not self._recording or self._recording_plotter is None:
            return
        try:
            self._recording_plotter.write_frame()
        except Exception:
            # Writer failed mid-recording — stop gracefully.
            self._stop_recording()
            self._uncheck_record()

    # ── Overlays ──────────────────────────────────────────────────────────────

    def _toggle_axes(self, checked: bool):
        p = self._plotter
        if p is None:
            return
        self._show_axes = checked
        try:
            if checked:
                p.show_axes()
            else:
                p.hide_axes()
            p.render()
        except Exception:
            pass

    def _toggle_bbox(self, checked: bool):
        p = self._plotter
        if p is None:
            return
        self._show_bbox = checked
        try:
            if checked:
                p.add_bounding_box()
            else:
                p.remove_bounding_box()
            p.render()
        except Exception:
            pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _btn(
        self,
        text: str,
        icon_name: str,
        tooltip: str,
        *,
        checkable: bool = False,
    ) -> QtWidgets.QToolButton:
        btn = QtWidgets.QToolButton()
        ic = _icon(icon_name)
        if ic is not None:
            btn.setIcon(ic)
            btn.setIconSize(QtCore.QSize(18, 18))
            if text:
                btn.setText(text)
                btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            else:
                btn.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        else:
            _FALLBACK = {
                "mdi.rotate-3d-variant":       "⬡",
                "mdi.arrow-collapse-up":        "⬆",
                "mdi.arrow-collapse-down":      "⬇",
                "mdi.arrow-expand-up":          "↑",
                "mdi.arrow-expand-down":        "↓",
                "mdi.arrow-expand-left":        "←",
                "mdi.arrow-expand-right":       "→",
                "mdi.fit-to-page-outline":      "⌖",
                "mdi.grid":                     "⊞",
                "mdi.camera-outline":           "📷",
                "mdi.record-circle-outline":    "⏺",
                "mdi.axis-arrow":               "✛",
                "mdi.crop-square":              "▭",
            }
            btn.setText(text or _FALLBACK.get(icon_name, icon_name.split(".")[-1]))
        btn.setToolTip(tooltip)
        btn.setAutoRaise(True)
        btn.setCheckable(checkable)
        btn.setFixedHeight(26)
        return btn

    @staticmethod
    def _add_sep(layout: QtWidgets.QHBoxLayout):
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.VLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        sep.setFixedWidth(8)
        layout.addWidget(sep)
