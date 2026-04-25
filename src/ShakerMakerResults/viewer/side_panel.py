"""Right-side panel for the interactive viewer.

Layout
------
A narrow vertical nav-menu (84 px) selects pages in a ``QStackedWidget``.
Only the active page is ever visible.

Performance contract
--------------------
Lightweight pages (Node → Warp)
    Built at startup.  Refreshed unconditionally on every ``refresh()``
    call except ``"time"`` — they are pure Qt-label / combo updates.

Heavy pages (Traces, Spectrum, Arias — all Matplotlib canvases)
    Built lazily on first navigation.  Refreshed **only** when active.

    *Dirty tracking*: when a ``"selection"`` or ``"full"`` update arrives
    while a heavy page is *not* active, that page is flagged dirty.  The
    next time the user navigates to it, it receives a ``"full"`` refresh
    before becoming visible, so it always shows current data.

    This means Spectrum and Arias never compute Newmark or Arias-intensity
    unless the user actually visits those pages.

``"time"`` updates
    Only the active heavy page (trace cursors) is touched, throttled to
    one redraw per 120 ms during playback.
"""

from __future__ import annotations

import time

from ._imports import require_viewer_dependencies
from .colors import COLORMAP_OPTIONS
from .trace_panel import AriasIntensityPanel, CombinedTracePanel, SpectrumPanel

_, _, _, _, _, QtWidgets = require_viewer_dependencies()


# ── Menu configuration ────────────────────────────────────────────────────────

_MENU_ENTRIES: list[tuple[str, str]] = [
    ("Node",       "Node"),
    ("Data",       "Data"),
    ("Colors",     "Colors"),
    ("Visibility", "Visibility"),
    ("Warp",       "Warp"),
    # ── separator ── analysis pages below ──
    ("Traces",     "Traces"),
    ("Spectrum",   "Spectrum"),
    ("Arias",      "Arias"),
]

# Keys whose pages are heavy (lazy-created; refresh only when active).
_HEAVY_KEYS: frozenset[str] = frozenset({"Traces", "Spectrum", "Arias"})

# Reasons that make a heavy page's content stale if it wasn't active.
_STALE_REASONS: frozenset[str] = frozenset({"selection", "full"})

_TRACE_TIME_REFRESH_INTERVAL_S = 0.08


# ── Lazy page wrapper ─────────────────────────────────────────────────────────

class _LazyPage(QtWidgets.QWidget):
    """Delays construction of a heavy widget until first navigation.

    ``ensure_created()`` triggers the factory and embeds the result.
    ``refresh()`` is a no-op until the widget is created.
    """

    def __init__(self, factory, parent=None):
        super().__init__(parent)
        self._factory = factory
        self._widget: QtWidgets.QWidget | None = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._lay = lay

    def ensure_created(self) -> QtWidgets.QWidget:
        if self._widget is None:
            self._widget = self._factory()
            self._lay.addWidget(self._widget)
        return self._widget

    def refresh(self, reason: str):
        if self._widget is not None:
            self._widget.refresh(reason)

    @property
    def is_created(self) -> bool:
        return self._widget is not None


# ── Shared base for Apply-button sections ─────────────────────────────────────

class _SectionBase(QtWidgets.QWidget):

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._syncing = False
        self._dirty = False
        self.apply_button: QtWidgets.QPushButton | None = None

    def _make_apply_button(self, callback) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton("Apply")
        btn.setEnabled(False)
        btn.clicked.connect(callback)
        self.apply_button = btn
        return btn

    def _set_dirty(self, dirty: bool = True):
        if self._syncing:
            return
        self._dirty = bool(dirty)
        if self.apply_button is not None:
            self.apply_button.setEnabled(self._dirty)

    def _clear_dirty(self):
        self._dirty = False
        if self.apply_button is not None:
            self.apply_button.setEnabled(False)

    @staticmethod
    def _set_combo(combo, value):
        b = combo.blockSignals(True)
        combo.setCurrentText(str(value))
        combo.blockSignals(b)

    @staticmethod
    def _coord_spin() -> QtWidgets.QDoubleSpinBox:
        s = QtWidgets.QDoubleSpinBox()
        s.setDecimals(3)
        s.setRange(-1e12, 1e12)
        s.setSingleStep(1.0)
        return s

    @staticmethod
    def _value_spin() -> QtWidgets.QDoubleSpinBox:
        s = QtWidgets.QDoubleSpinBox()
        s.setDecimals(8)
        s.setRange(-1e20, 1e20)
        s.setSingleStep(0.1)
        return s

    @staticmethod
    def _fmt_xyz(xyz) -> str:
        return "-" if xyz is None else f"{xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}"


# ── Lightweight section widgets ───────────────────────────────────────────────

class NodeSearchSection(QtWidgets.QWidget):
    """Selected-node info + coordinate search."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._syncing = False

        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(6, 6, 6, 6)

        self.node_id_spin = QtWidgets.QSpinBox()
        self.node_id_spin.setMinimum(0)
        self.node_id_spin.setMaximum(max(len(session.adapter.node_ids) - 1, 0))
        self.node_select_btn = QtWidgets.QPushButton("Select")
        self.node_select_btn.clicked.connect(self._select_by_id)

        node_row = QtWidgets.QWidget()
        nrl = QtWidgets.QHBoxLayout(node_row)
        nrl.setContentsMargins(0, 0, 0, 0)
        nrl.addWidget(self.node_id_spin, 1)
        nrl.addWidget(self.node_select_btn)

        self.x_spin = _SectionBase._coord_spin()
        self.y_spin = _SectionBase._coord_spin()
        self.z_spin = _SectionBase._coord_spin()
        self.find_btn = QtWidgets.QPushButton("Find nearest")
        self.find_btn.clicked.connect(self._find_nearest)
        self.nearest_lbl = QtWidgets.QLabel("-")

        self.node_lbl      = QtWidgets.QLabel("None")
        self.type_lbl      = QtWidgets.QLabel("-")
        self.xyz_model_lbl = QtWidgets.QLabel("-")
        self.xyz_view_lbl  = QtWidgets.QLabel("-")
        self.gf_slot_lbl   = QtWidgets.QLabel("-")

        form.addRow("Node ID",       node_row)
        form.addRow("X [m]",         self.x_spin)
        form.addRow("Y [m]",         self.y_spin)
        form.addRow("Z [m]",         self.z_spin)
        form.addRow("",              self.find_btn)
        form.addRow("Nearest",       self.nearest_lbl)
        form.addRow("Selected",      self.node_lbl)
        form.addRow("Type",          self.type_lbl)
        form.addRow("XYZ model [m]", self.xyz_model_lbl)
        form.addRow("XYZ view [m]",  self.xyz_view_lbl)
        form.addRow("GF slot s0",    self.gf_slot_lbl)

        self.refresh("init")

    def refresh(self, _reason: str = "full"):
        self._syncing = True
        try:
            info = self.session.current_node_info()
            if info is None:
                self.node_lbl.setText("None")
                for lbl in (self.type_lbl, self.xyz_model_lbl,
                            self.xyz_view_lbl, self.gf_slot_lbl):
                    lbl.setText("-")
                return
            nid = info["node_id"]
            if nid not in ("QA", "qa"):
                self.node_id_spin.setValue(int(nid))
            xyz_m = info.get("xyz_model_m")
            self.node_lbl.setText(str(nid))
            self.type_lbl.setText(str(info["type"]))
            self.xyz_model_lbl.setText(_SectionBase._fmt_xyz(xyz_m))
            self.xyz_view_lbl.setText(_SectionBase._fmt_xyz(info.get("xyz_m")))
            self.gf_slot_lbl.setText(
                "-" if info["gf_slot_s0"] is None else str(info["gf_slot_s0"])
            )
            if xyz_m is not None:
                self.x_spin.setValue(float(xyz_m[0]))
                self.y_spin.setValue(float(xyz_m[1]))
                self.z_spin.setValue(float(xyz_m[2]))
        finally:
            self._syncing = False

    def _select_by_id(self):
        if not self._syncing:
            self.session.select_node(int(self.node_id_spin.value()))

    def _find_nearest(self):
        if self._syncing:
            return
        nid, dist = self.session.select_nearest_coordinate_m(
            self.x_spin.value(), self.y_spin.value(), self.z_spin.value()
        )
        self.nearest_lbl.setText(f"{nid} | dist={dist:.3f} m")


class DataSection(_SectionBase):
    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        form = QtWidgets.QFormLayout()
        self.demand_combo = QtWidgets.QComboBox()
        for d in session.adapter.available_demands:
            self.demand_combo.addItem(d, d)
        self.demand_combo.currentTextChanged.connect(lambda _: self._set_dirty())

        self.component_combo = QtWidgets.QComboBox()
        for c in session.adapter.available_components:
            self.component_combo.addItem(c, c)
        self.component_combo.currentTextChanged.connect(lambda _: self._set_dirty())

        form.addRow("Demand",    self.demand_combo)
        form.addRow("Component", self.component_combo)
        outer.addLayout(form)
        outer.addWidget(self._make_apply_button(self._apply))
        outer.addStretch(1)
        self.refresh("init")

    def refresh(self, reason: str = "full"):
        if reason not in {"init", "full", "demand", "component"} and self._dirty:
            return
        self._syncing = True
        try:
            self._set_combo(self.demand_combo, self.session.state.demand)
            self._set_combo(self.component_combo, self.session.state.component)
            self._clear_dirty()
        finally:
            self._syncing = False

    def _apply(self):
        if self._syncing:
            return
        self.session.apply_data_settings(
            demand=self.demand_combo.currentText(),
            component=self.component_combo.currentText(),
        )
        self._clear_dirty()


class ColorMapSection(_SectionBase):
    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        form = QtWidgets.QFormLayout()
        self.cmap_combo = QtWidgets.QComboBox()
        for cmap in COLORMAP_OPTIONS:
            self.cmap_combo.addItem(cmap, cmap)
        self.cmap_combo.currentTextChanged.connect(lambda *_: self._set_dirty())

        self.vmin_spin = self._value_spin()
        self.vmax_spin = self._value_spin()
        self.vmin_spin.valueChanged.connect(lambda *_: self._set_dirty())
        self.vmax_spin.valueChanged.connect(lambda *_: self._set_dirty())

        self.clamp_cb = QtWidgets.QCheckBox("Clamp to range")
        self.clamp_cb.toggled.connect(lambda *_: self._set_dirty())

        self.reset_btn = QtWidgets.QPushButton("Reset to auto")
        self.reset_btn.clicked.connect(self._reset_range)
        self.auto_lbl = QtWidgets.QLabel("-")

        form.addRow("Preset",     self.cmap_combo)
        form.addRow("Auto range", self.auto_lbl)
        form.addRow("User min",   self.vmin_spin)
        form.addRow("User max",   self.vmax_spin)
        form.addRow("",           self.clamp_cb)
        form.addRow("",           self.reset_btn)
        outer.addLayout(form)
        outer.addWidget(self._make_apply_button(self._apply))
        outer.addStretch(1)
        self.refresh("init")

    def refresh(self, reason: str = "full"):
        if reason not in {
            "init", "full", "appearance", "color_range", "demand", "component"
        } and self._dirty:
            return
        self._syncing = True
        try:
            self._set_combo(self.cmap_combo, self.session.current_colormap())
            lo, hi = self.session.adapter.default_scalar_limits(
                self.session.state.demand, self.session.state.component
            )
            self.auto_lbl.setText(f"{lo:.4g}  /  {hi:.4g}")
            if self.session.state.user_vmin is None or self.session.state.user_vmax is None:
                self.session.state.set_user_color_range(lo, hi)
            self.vmin_spin.setValue(float(self.session.state.user_vmin))
            self.vmax_spin.setValue(float(self.session.state.user_vmax))
            self.clamp_cb.setChecked(self.session.state.clamp_enabled)
            self._clear_dirty()
        finally:
            self._syncing = False

    def _reset_range(self):
        lo, hi = self.session.adapter.default_scalar_limits(
            self.session.state.demand, self.session.state.component
        )
        for spin, val in ((self.vmin_spin, lo), (self.vmax_spin, hi)):
            b = spin.blockSignals(True); spin.setValue(val); spin.blockSignals(b)
        b = self.clamp_cb.blockSignals(True)
        self.clamp_cb.setChecked(False)
        self.clamp_cb.blockSignals(b)
        self._set_dirty(True)

    def _apply(self):
        if self._syncing:
            return
        self.session.apply_color_settings(
            colormap=self.cmap_combo.currentText(),
            vmin=self.vmin_spin.value(),
            vmax=self.vmax_spin.value(),
            clamp_enabled=self.clamp_cb.isChecked(),
        )
        self._clear_dirty()


class VisualizationSection(_SectionBase):
    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        self.internal_cb = QtWidgets.QCheckBox("Show internal nodes")
        self.external_cb = QtWidgets.QCheckBox("Show external nodes")
        self.qa_cb       = QtWidgets.QCheckBox("Show QA")
        for cb in (self.internal_cb, self.external_cb, self.qa_cb):
            cb.toggled.connect(lambda *_: self._set_dirty())
            lay.addWidget(cb)

        lay.addWidget(self._make_apply_button(self._apply))
        lay.addStretch(1)
        self.refresh("init")

    def refresh(self, reason: str = "full"):
        if reason not in {"init", "full", "visibility"} and self._dirty:
            return
        self._syncing = True
        try:
            self.internal_cb.setChecked(self.session.state.show_internal)
            self.external_cb.setChecked(self.session.state.show_external)
            self.qa_cb.setChecked(self.session.state.show_qa)
            self._clear_dirty()
        finally:
            self._syncing = False

    def _apply(self):
        if self._syncing:
            return
        self.session.apply_visibility_settings(
            show_internal=self.internal_cb.isChecked(),
            show_external=self.external_cb.isChecked(),
            show_qa=self.qa_cb.isChecked(),
        )
        self._clear_dirty()


class WarpSection(_SectionBase):
    def __init__(self, session, parent=None):
        super().__init__(session, parent)
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        self.warp_cb = QtWidgets.QCheckBox("Enable displacement warp")
        self.warp_cb.toggled.connect(lambda *_: self._set_dirty())
        outer.addWidget(self.warp_cb)

        axes_row = QtWidgets.QWidget()
        arl = QtWidgets.QHBoxLayout(axes_row)
        arl.setContentsMargins(0, 0, 0, 0)
        arl.setSpacing(6)
        arl.addWidget(QtWidgets.QLabel("Axes:"))
        self.x_cb = QtWidgets.QCheckBox("X (E)")
        self.y_cb = QtWidgets.QCheckBox("Y (N)")
        self.z_cb = QtWidgets.QCheckBox("Z")
        for cb in (self.x_cb, self.y_cb, self.z_cb):
            cb.setChecked(True)
            cb.toggled.connect(lambda *_: self._set_dirty())
            arl.addWidget(cb)
        arl.addStretch()
        outer.addWidget(axes_row)

        scale_row = QtWidgets.QWidget()
        srl = QtWidgets.QHBoxLayout(scale_row)
        srl.setContentsMargins(0, 0, 0, 0)
        srl.setSpacing(4)
        srl.addWidget(QtWidgets.QLabel("Scale:"))
        for lbl, val in (("×10", 10), ("×100", 100), ("×1k", 1_000), ("×10k", 10_000)):
            btn = QtWidgets.QPushButton(lbl)
            btn.setFixedWidth(40)
            btn.clicked.connect(lambda _c, v=val: self._preset(v))
            srl.addWidget(btn)
        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.scale_spin.setDecimals(1)
        self.scale_spin.setRange(1.0, 1_000_000.0)
        self.scale_spin.setSingleStep(10.0)
        self.scale_spin.setValue(100.0)
        self.scale_spin.setSuffix("×")
        self.scale_spin.valueChanged.connect(lambda *_: self._set_dirty())
        srl.addWidget(self.scale_spin, 1)
        auto_btn = QtWidgets.QPushButton("Auto")
        auto_btn.setToolTip("Suggest scale so peak displacement fills ~5 % of domain.")
        auto_btn.clicked.connect(self._auto)
        srl.addWidget(auto_btn)
        outer.addWidget(scale_row)

        outer.addWidget(self._make_apply_button(self._apply))
        outer.addStretch(1)
        self.refresh("init")

    def refresh(self, reason: str = "full"):
        if reason not in {"init", "full", "warp"} and self._dirty:
            return
        self._syncing = True
        try:
            self.warp_cb.setChecked(self.session.state.disp_warp_enabled)
            ax = self.session.state.warp_axes
            self.x_cb.setChecked(ax[0])
            self.y_cb.setChecked(ax[1])
            self.z_cb.setChecked(ax[2])
            scale = (
                float(self.session.state.warp_scale)
                if self.session.state.warp_scale is not None
                else float(self.session.suggested_warp_scale())
            )
            self.scale_spin.setValue(scale)
            self._clear_dirty()
        finally:
            self._syncing = False

    def _preset(self, val: float):
        if self._syncing:
            return
        b = self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(float(val))
        self.scale_spin.blockSignals(b)
        self._set_dirty(True)

    def _auto(self):
        b = self.scale_spin.blockSignals(True)
        self.scale_spin.setValue(float(self.session.suggested_warp_scale()))
        self.scale_spin.blockSignals(b)
        self._set_dirty(True)

    def _apply(self):
        if self._syncing:
            return
        self.session.apply_warp_settings(
            warp_enabled=self.warp_cb.isChecked(),
            warp_axes=(self.x_cb.isChecked(), self.y_cb.isChecked(), self.z_cb.isChecked()),
            warp_scale=self.scale_spin.value(),
        )
        self._clear_dirty()


# ── Main side panel ───────────────────────────────────────────────────────────

class ViewerSidePanel(QtWidgets.QWidget):
    """Vertical nav-menu + ``QStackedWidget`` right panel.

    Menu: Node | Data | Colors | Visibility | Warp || Traces | Spectrum | Arias
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._active_key: str = "Node"
        self._last_trace_refresh_at: float = 0.0

        # Heavy pages that missed a stale-making update while not active.
        self._dirty_heavy: set[str] = set()

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Vertical nav menu ─────────────────────────────────────────────────
        nav = QtWidgets.QWidget()
        nav.setFixedWidth(84)
        nav_lay = QtWidgets.QVBoxLayout(nav)
        nav_lay.setContentsMargins(0, 4, 0, 4)
        nav_lay.setSpacing(1)

        self._nav_buttons: dict[str, QtWidgets.QPushButton] = {}
        group = QtWidgets.QButtonGroup(self)
        group.setExclusive(True)

        for key, label in _MENU_ENTRIES:
            if key == "Traces":
                sep = QtWidgets.QFrame()
                sep.setFrameShape(QtWidgets.QFrame.HLine)
                sep.setFixedHeight(1)
                sep.setStyleSheet("background: #d7dbe2;")
                nav_lay.addSpacing(4)
                nav_lay.addWidget(sep)
                nav_lay.addSpacing(4)

            btn = QtWidgets.QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            btn.clicked.connect(lambda _c, k=key: self._navigate(k))
            self._nav_buttons[key] = btn
            group.addButton(btn)
            nav_lay.addWidget(btn)

        nav_lay.addStretch(1)
        nav.setStyleSheet(
            "QWidget { background: #f3f4f6; border-right: 1px solid #d7dbe2; }"
            "QPushButton {"
            "  border: none; border-radius: 0;"
            "  text-align: left; padding: 4px 10px;"
            "  background: transparent; color: #222; font-size: 12px;"
            "}"
            "QPushButton:checked { background: #1565C0; color: white; }"
            "QPushButton:hover:!checked { background: #e3eaf6; }"
        )
        root.addWidget(nav)

        # ── Content area ──────────────────────────────────────────────────────
        self._stack = QtWidgets.QStackedWidget()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(self._stack)
        root.addWidget(scroll, 1)

        # ── Build pages ───────────────────────────────────────────────────────
        self._node_page       = NodeSearchSection(session)
        self._data_page       = DataSection(session)
        self._colors_page     = ColorMapSection(session)
        self._visibility_page = VisualizationSection(session)
        self._warp_page       = WarpSection(session)

        # Heavy pages: created lazily on first navigation.
        self._traces_page   = _LazyPage(lambda: CombinedTracePanel(session))
        self._spectrum_page = _LazyPage(lambda: SpectrumPanel(session))
        self._arias_page    = _LazyPage(lambda: AriasIntensityPanel(session))

        self._pages: dict[str, QtWidgets.QWidget] = {
            "Node":       self._node_page,
            "Data":       self._data_page,
            "Colors":     self._colors_page,
            "Visibility": self._visibility_page,
            "Warp":       self._warp_page,
            "Traces":     self._traces_page,
            "Spectrum":   self._spectrum_page,
            "Arias":      self._arias_page,
        }

        for page in self._pages.values():
            self._stack.addWidget(page)

        self._navigate("Node")

    # ── Navigation ────────────────────────────────────────────────────────────

    def _navigate(self, key: str):
        self._active_key = key

        btn = self._nav_buttons.get(key)
        if btn is not None:
            btn.setChecked(True)

        page = self._pages[key]

        if isinstance(page, _LazyPage):
            already_existed = page.is_created
            page.ensure_created()
            # If the page existed but missed a selection/full update, catch up now.
            if already_existed and key in self._dirty_heavy:
                page.refresh("full")
        elif key in self._dirty_heavy:
            page.refresh("full")

        self._dirty_heavy.discard(key)
        self._stack.setCurrentWidget(page)

    # ── Refresh routing ───────────────────────────────────────────────────────

    def refresh(self, reason: str):
        """Route a session-update reason to the correct sub-widgets.

        Only the active page is refreshed for heavy content.  Non-active heavy
        pages that depend on ``reason`` are flagged dirty and will refresh the
        next time the user navigates to them.
        """
        # ── Time: cursor-only update on the active heavy page ─────────────────
        # CombinedTracePanel uses matplotlib blitting (< 1 ms per update), so
        # no throttle is needed for Traces.  Spectrum and Arias are no-ops on
        # "time", so calling them unconditionally is free.
        if reason == "time":
            if self._active_key in _HEAVY_KEYS:
                if self._active_key == "Traces" and self.session.state.is_playing:
                    now = time.perf_counter()
                    if (now - self._last_trace_refresh_at) < _TRACE_TIME_REFRESH_INTERVAL_S:
                        return
                    self._last_trace_refresh_at = now
                page = self._pages.get(self._active_key)
                if page is not None:
                    page.refresh("time")
            return

        if reason == "playback":
            if not self.session.state.is_playing and self._active_key == "Traces":
                page = self._pages.get("Traces")
                if page is not None:
                    page.refresh("time")
            return

        # ── All other reasons: lightweight pages always, heavy only if active ──

        # Lightweight pages — cheap Qt-label updates, always safe to refresh.
        self._node_page.refresh(reason)
        self._data_page.refresh(reason)
        self._colors_page.refresh(reason)
        self._visibility_page.refresh(reason)
        self._warp_page.refresh(reason)

        # Mark non-active heavy pages dirty when their content would change.
        if reason in _STALE_REASONS:
            for key in _HEAVY_KEYS:
                if key != self._active_key:
                    self._dirty_heavy.add(key)

        # Refresh the active heavy page immediately.
        if self._active_key in _HEAVY_KEYS:
            page = self._pages.get(self._active_key)
            if page is not None:
                page.refresh(reason)
                self._dirty_heavy.discard(self._active_key)


__all__ = ["ViewerSidePanel"]
