"""Multi-viewport central area for the interactive viewer.

Provides :class:`ViewPane` (one VTK viewport) and :class:`MultiViewArea`
(a layout-selector bar + a dynamic grid of panes).

Design contract
---------------
* A single :class:`~.session.ViewerSession` is shared by every pane — all
  state (time, demand, selection, warp …) is global.
* Each pane owns an independent :class:`~pyvistaqt.QtInteractor` and a
  :class:`~.scene.ViewerScene`, so cameras and orientations are per-pane.
* On every ``on_session_updated(reason)`` call from the window, the
  ``MultiViewArea`` broadcasts the reason to every visible pane; each pane
  calls the matching scene method and then ``plotter.render()``.
* Clicking anywhere in a pane marks it as *active* (thin blue top-bar).
  The toolbar's view-preset and capture operations act on the active pane.
"""

from __future__ import annotations

from ._imports import require_viewer_dependencies
from .scene import ViewerScene

_, QtInteractor, _, QtCore, _, QtWidgets = require_viewer_dependencies()


# ── Constants ─────────────────────────────────────────────────────────────────

#: Default label assigned to new panes in creation order (used for initial
#: camera orientation only — there is no visible combo in the pane UI).
_PANE_DEFAULTS: list[str] = [
    "3D NE", "Top", "Front", "Right", "3D NW", "3D SW", "3D SE",
]

#: (layout name, number of panes required) — order matches the toolbar.
LAYOUT_PRESETS: list[tuple[str, int]] = [
    ("1×1", 1),
    ("1×2", 2),
    ("2×1", 2),
    ("2×2", 4),
    ("2+1", 3),
    ("1+2", 3),
]

# Azimuth offsets for ISO presets (degrees, applied after view_isometric).
_ISO_AZIMUTH: dict[str, int] = {
    "3D NE": 0,
    "3D NW": 90,
    "3D SW": 180,
    "3D SE": 270,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

class _ClickFilter(QtCore.QObject):
    """Event-filter that fires *callback* on any ``MouseButtonPress`` event."""

    def __init__(self, callback, parent=None):
        super().__init__(parent)
        self._cb = callback

    def eventFilter(self, obj, event):           # noqa: N802
        if event.type() == QtCore.QEvent.MouseButtonPress:
            self._cb()
        return False                              # never consume the event


# ── ViewPane ──────────────────────────────────────────────────────────────────

class ViewPane(QtWidgets.QWidget):
    """One viewport: 3 px active-indicator strip + full-height QtInteractor.

    No per-pane menu or combo is shown — camera presets are controlled
    exclusively from the toolbar above.

    Parameters
    ----------
    session:
        The shared viewer session (data + state).
    label:
        Initial camera orientation key (one of the ``_PANE_DEFAULTS``).
        Applied once at construction; not displayed.
    on_activated:
        Callable ``(pane: ViewPane) -> None`` fired when the user clicks
        anywhere in this pane.  Used by :class:`MultiViewArea` to track the
        active pane.
    """

    def __init__(
        self,
        session,
        label: str = "3D NE",
        on_activated=None,
        parent=None,
    ):
        super().__init__(parent)
        self.session = session
        self._on_activated = on_activated
        self._is_active = False

        self.setMinimumSize(80, 80)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Active-pane indicator (thin colored strip at the top) ─────────────
        self._indicator = QtWidgets.QWidget()
        self._indicator.setFixedHeight(3)
        self._indicator.setVisible(False)
        outer.addWidget(self._indicator)

        # ── VTK interactor ───────────────────────────────────────────────────
        self.plotter = QtInteractor(self)
        outer.addWidget(self.plotter.interactor, 1)

        # ── Scene ────────────────────────────────────────────────────────────
        self.scene = ViewerScene(self.plotter, session)
        try:
            self.scene.build()
            self._apply_initial_camera(label)
        except Exception:
            pass

        # ── Click → activate ─────────────────────────────────────────────────
        self._click_filter = _ClickFilter(self._fire_activated, self)
        self.plotter.interactor.installEventFilter(self._click_filter)

    # ── Camera ────────────────────────────────────────────────────────────────

    def _apply_initial_camera(self, label: str):
        """Set the camera to the position described by *label* at startup."""
        p = self.plotter
        try:
            if label in _ISO_AZIMUTH:
                p.view_isometric()
                az = _ISO_AZIMUTH[label]
                if az:
                    p.camera.Azimuth(az)
                    p.reset_camera_clipping_range()
            elif label == "Top":
                p.view_xy()
            elif label == "Bottom":
                p.view_xy(negative=True)
            elif label == "Front":
                p.view_xz()
            elif label == "Back":
                p.view_xz(negative=True)
            elif label == "Left":
                p.view_yz()
            elif label == "Right":
                p.view_yz(negative=True)
            p.render()
        except Exception:
            pass

    # ── Active-pane highlight ─────────────────────────────────────────────────

    def _fire_activated(self):
        if self._on_activated is not None:
            self._on_activated(self)

    def set_active(self, active: bool):
        if self._is_active == active:
            return
        self._is_active = active
        self._indicator.setVisible(active)
        if active:
            self._indicator.setStyleSheet("background: #1565C0;")

    # ── Scene refresh ─────────────────────────────────────────────────────────

    def on_session_updated(self, reason: str):
        """Apply *reason* to this pane's scene and re-render."""
        if reason == "time":
            self.scene.refresh_scalars(render=False)
        elif reason == "selection":
            self.scene.refresh_selection(render=False)
        elif reason in {"demand", "component"}:
            self.scene.rebuild_scalar_actor(render=False)
            self.scene.refresh_selection(render=False)
        elif reason == "visibility":
            self.scene.rebuild_for_visibility(render=False)
        elif reason == "color_range":
            self.scene.apply_color_range(render=False)
        elif reason in {"appearance", "panel_apply"}:
            self.scene.apply_appearance(render=False)
            self.scene.refresh_selection(render=False)
        elif reason == "warp":
            self.scene.rebuild_scalar_actor(render=False)
            self.scene.refresh_selection(render=False)
        else:
            # "full" or any unknown reason — full rebuild.
            self.scene.rebuild_scalar_actor(render=False)
            self.scene.refresh_selection(render=False)
        try:
            self.plotter.render()
        except Exception:
            pass


# ── MultiViewArea ─────────────────────────────────────────────────────────────

class MultiViewArea(QtWidgets.QWidget):
    """Central zone that manages 1–N synchronised :class:`ViewPane` instances.

    The layout is controlled by a compact selector bar at the top.  Switching
    layouts reparents existing panes into a new splitter tree without destroying
    them, so cameras and rendering state are preserved across layout changes.

    Parameters
    ----------
    session:
        The shared viewer session passed to every new :class:`ViewPane`.
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._panes: list[ViewPane] = []
        self._active_pane: ViewPane | None = None
        self._current_layout: str = "1×1"
        self._container: QtWidgets.QWidget | None = None
        self._layout_n: dict[str, int] = dict(LAYOUT_PRESETS)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Layout selector bar ───────────────────────────────────────────────
        bar = QtWidgets.QWidget()
        bar.setFixedHeight(30)
        bar.setStyleSheet("background: #f0f0f0; border-bottom: 1px solid #d0d0d0;")
        blay = QtWidgets.QHBoxLayout(bar)
        blay.setContentsMargins(6, 3, 6, 3)
        blay.setSpacing(3)

        lbl = QtWidgets.QLabel("Layout:")
        lbl.setStyleSheet("color: #404040; font-size: 11px;")
        blay.addWidget(lbl)

        self._layout_buttons: dict[str, QtWidgets.QPushButton] = {}
        for name, _n in LAYOUT_PRESETS:
            btn = QtWidgets.QPushButton(name)
            btn.setFixedSize(44, 22)
            btn.setCheckable(True)
            btn.setStyleSheet(
                "QPushButton { border: 1px solid #bbb; border-radius: 3px;"
                "  background: #fff; font-size: 11px; }"
                "QPushButton:checked { background: #1565C0; color: white;"
                "  border-color: #1565C0; }"
                "QPushButton:hover:!checked { background: #e3eaf6; }"
            )
            btn.clicked.connect(lambda _c, n=name: self._switch_layout(n))
            self._layout_buttons[name] = btn
            blay.addWidget(btn)

        blay.addStretch(1)
        root.addWidget(bar)

        # ── Content area ──────────────────────────────────────────────────────
        self._content = QtWidgets.QWidget()
        self._content_lay = QtWidgets.QVBoxLayout(self._content)
        self._content_lay.setContentsMargins(0, 0, 0, 0)
        self._content_lay.setSpacing(0)
        root.addWidget(self._content, 1)

        # Start with a single pane.
        self._switch_layout("1×1")

    # ── Layout switching ──────────────────────────────────────────────────────

    def _switch_layout(self, name: str):
        """Switch to layout *name*, creating panes as needed."""
        needed = self._layout_n.get(name, 1)

        # 1. Detach all panes from the current container (reparent, not delete).
        for pane in self._panes:
            pane.setParent(self)
            pane.hide()

        # 2. Remove and schedule the old container for deletion.
        if self._container is not None:
            self._content_lay.removeWidget(self._container)
            self._container.setParent(None)
            self._container.deleteLater()
            self._container = None

        # 3. Create additional panes if the layout needs more than we have.
        while len(self._panes) < needed:
            label = _PANE_DEFAULTS[len(self._panes) % len(_PANE_DEFAULTS)]
            pane = ViewPane(
                self.session,
                label=label,
                on_activated=self._on_pane_activated,
                parent=self,
            )
            self._panes.append(pane)

        # 4. Build the new splitter/container.
        panes_for_layout = self._panes[:needed]
        self._container = self._build_container(name, panes_for_layout)
        self._content_lay.addWidget(self._container)
        self._container.show()

        # addWidget already reparented each pane into the container.
        for pane in panes_for_layout:
            pane.show()

        # 5. Ensure a valid active pane.
        if self._active_pane not in panes_for_layout:
            self._set_active_pane(panes_for_layout[0] if panes_for_layout else None)

        # 6. Sync button states.
        self._current_layout = name
        for n, btn in self._layout_buttons.items():
            btn.setChecked(n == name)

    def _build_container(
        self, name: str, panes: list[ViewPane]
    ) -> QtWidgets.QWidget:
        """Assemble a splitter tree for *name* using the given pane list."""

        def _h(*ps) -> QtWidgets.QSplitter:
            s = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            s.setChildrenCollapsible(False)
            for p in ps:
                s.addWidget(p)
            return s

        def _v(*ps) -> QtWidgets.QSplitter:
            s = QtWidgets.QSplitter(QtCore.Qt.Vertical)
            s.setChildrenCollapsible(False)
            for p in ps:
                s.addWidget(p)
            return s

        if name == "1×1":
            w = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(w)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(panes[0])
            return w

        if name == "1×2":
            return _h(panes[0], panes[1])

        if name == "2×1":
            return _v(panes[0], panes[1])

        if name == "2×2":
            return _v(_h(panes[0], panes[1]), _h(panes[2], panes[3]))

        if name == "2+1":
            # Two panes on the top row, one full-width on the bottom.
            outer = _v(_h(panes[0], panes[1]), panes[2])
            outer.setSizes([600, 300])
            return outer

        if name == "1+2":
            # One full-width pane on top, two panes on the bottom row.
            outer = _v(panes[0], _h(panes[1], panes[2]))
            outer.setSizes([300, 600])
            return outer

        # Fallback — plain horizontal split.
        return _h(*panes)

    # ── Active pane ───────────────────────────────────────────────────────────

    def _on_pane_activated(self, pane: ViewPane):
        self._set_active_pane(pane)

    def _set_active_pane(self, pane: ViewPane | None):
        for p in self._panes:
            p.set_active(p is pane)
        self._active_pane = pane

    @property
    def active_pane(self) -> ViewPane | None:
        """The pane the user last clicked, or the first pane."""
        return self._active_pane

    @property
    def active_plotter(self):
        """Plotter of the active pane — used by the toolbar for capture/views."""
        if self._active_pane is not None:
            return self._active_pane.plotter
        return self._panes[0].plotter if self._panes else None

    # ── Session broadcast ─────────────────────────────────────────────────────

    def on_session_updated(self, reason: str):
        """Broadcast *reason* to every pane that is currently visible."""
        needed = self._layout_n.get(self._current_layout, 1)
        for pane in self._panes[:needed]:
            pane.on_session_updated(reason)
