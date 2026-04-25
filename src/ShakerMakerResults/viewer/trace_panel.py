"""Matplotlib-based analysis panels used by the interactive viewer."""

from __future__ import annotations

from ._imports import require_viewer_dependencies

_, _, _, _, _, QtWidgets = require_viewer_dependencies()

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:  # pragma: no cover - compatibility fallback
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


COMPONENT_COLORS = {
    "Z": "tab:blue",
    "E": "tab:orange",
    "N": "tab:green",
    "z": "tab:blue",
    "e": "tab:orange",
    "n": "tab:green",
}


class TracePanel(QtWidgets.QWidget):
    """Embedded matplotlib panel showing node traces."""

    def __init__(self, session, parent=None, *, demand: str | None = None, title: str | None = None):
        super().__init__(parent)
        self.session = session
        self.demand = None if demand is None else str(demand)
        self.fixed_title = title

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QtWidgets.QLabel("No node selected")
        self.figure = Figure(figsize=(5, 4), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.subplots(3, 1, sharex=True)
        self._trace_lines = []
        self._time_cursors = []
        self._current_key = None

        layout.addWidget(self.title_label)
        layout.addWidget(self.canvas, 1)

        self.refresh("init")

    def refresh(self, reason: str = "full"):
        if reason == "time" and self._current_key is not None:
            self._update_time_cursor()
            self.canvas.draw_idle()
            return

        node_id = self.session.state.selected_node
        if node_id is None:
            self._current_key = None
            self._trace_lines = []
            self._time_cursors = []
            for ax in self.axes:
                ax.clear()
            self.title_label.setText("No node selected")
            for ax, label in zip(self.axes, ("Z", "E", "N")):
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.25)
            self.axes[-1].set_xlabel("Time [s]")
            self.canvas.draw_idle()
            return

        demand = self.demand or self.session.state.demand
        trace = (
            self.session.adapter.trace(node_id, demand)
            if self.demand is not None
            else self.session.current_trace()
        )
        time = self.session.adapter.time
        labels = ("Z", "E", "N")
        node_key = (node_id, demand)

        if self._current_key != node_key:
            self._current_key = node_key
            self._trace_lines = []
            self._time_cursors = []
            for ax in self.axes:
                ax.clear()

            title = self.fixed_title or f"Node {node_id} | {demand} traces"
            self.title_label.setText(title)
            for ax, values, label in zip(self.axes, trace, labels):
                line, = ax.plot(
                    time,
                    values,
                    linewidth=1.2,
                    color=COMPONENT_COLORS[label],
                    label=label,
                )
                cursor = ax.axvline(self.session.current_time(), color="tab:red", alpha=0.35)
                self._trace_lines.append(line)
                self._time_cursors.append(cursor)
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.25)
                ax.legend(loc="upper right")
            self.axes[-1].set_xlabel("Time [s]")
        else:
            self.title_label.setText(self.fixed_title or f"Node {node_id} | {demand} traces")
            for line, values in zip(self._trace_lines, trace):
                line.set_ydata(values)

        self._update_time_cursor()
        self.canvas.draw_idle()

    def _update_time_cursor(self):
        current_time = self.session.current_time()
        for cursor in self._time_cursors:
            cursor.set_xdata([current_time, current_time])


class SpectrumPanel(QtWidgets.QWidget):
    """Panel that computes and shows Newmark PSa on demand."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._current_node = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QtWidgets.QLabel("No node selected")
        self.figure = Figure(figsize=(5, 4), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.subplots(3, 1, sharex=True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.canvas, 1)

        self.refresh("init")

    def refresh(self, reason: str = "full"):
        if reason == "time":
            return

        node_id = self.session.state.selected_node
        if node_id is None:
            self._current_node = None
            for ax in self.axes:
                ax.clear()
                ax.grid(True, alpha=0.25)
            self.title_label.setText("No node selected")
            self.axes[-1].set_xlabel("Period [s]")
            self.canvas.draw_idle()
            return

        if self._current_node == node_id and reason not in {"selection", "full", "init"}:
            return

        self._current_node = node_id
        for ax in self.axes:
            ax.clear()

        try:
            spectrum = self.session.current_spectrum()
        except Exception as exc:  # pragma: no cover - runtime-only when deps are missing
            self.title_label.setText(f"Spectrum unavailable: {exc}")
            for ax in self.axes:
                ax.grid(True, alpha=0.25)
            self.canvas.draw_idle()
            return

        self.title_label.setText(f"Node {node_id} | Newmark PSa")
        for ax, label in zip(self.axes, ("z", "e", "n")):
            ax.plot(
                spectrum["T"],
                spectrum[f"PSa_{label}"],
                linewidth=1.2,
                color=COMPONENT_COLORS[label],
                label=label.upper(),
            )
            ax.set_ylabel(label.upper())
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right")
        self.axes[-1].set_xlabel("Period [s]")
        self.canvas.draw_idle()


class AriasIntensityPanel(QtWidgets.QWidget):
    """Panel showing Arias intensity curves for the selected node."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._current_node = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.title_label = QtWidgets.QLabel("No node selected")
        self.figure = Figure(figsize=(5, 4), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.subplots(3, 1, sharex=True)

        layout.addWidget(self.title_label)
        layout.addWidget(self.canvas, 1)
        self.refresh("init")

    def refresh(self, reason: str = "full"):
        if reason == "time":
            return

        node_id = self.session.state.selected_node
        if node_id is None:
            self._current_node = None
            for ax in self.axes:
                ax.clear()
                ax.grid(True, alpha=0.25)
            self.title_label.setText("No node selected")
            self.axes[-1].set_xlabel("Time [s]")
            self.canvas.draw_idle()
            return

        if self._current_node == node_id and reason not in {"selection", "full", "init"}:
            return

        self._current_node = node_id
        for ax in self.axes:
            ax.clear()

        try:
            arias = self.session.current_arias()
        except Exception as exc:  # pragma: no cover - optional dependency/runtime
            self.title_label.setText(f"Arias unavailable: {exc}")
            for ax in self.axes:
                ax.grid(True, alpha=0.25)
            self.canvas.draw_idle()
            return

        time = arias["time"]
        self.title_label.setText(f"Node {node_id} | Arias Intensity")
        for ax, label in zip(self.axes, ("z", "e", "n")):
            item = arias["components"][label]
            ax.plot(
                time,
                item["IA_pct"],
                linewidth=1.2,
                color=COMPONENT_COLORS[label],
                label=f"{label.upper()} | Ia={item['ia_total']:.3f} m/s",
            )
            ax.axvline(item["t_start"], color=COMPONENT_COLORS[label], linestyle="--", linewidth=1, alpha=0.45)
            ax.axvline(item["t_end"], color=COMPONENT_COLORS[label], linestyle="--", linewidth=1, alpha=0.45)
            ax.axhline(5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
            ax.axhline(95, color="gray", linestyle=":", linewidth=1, alpha=0.6)
            ax.set_ylabel(label.upper())
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left")
        self.axes[-1].set_xlabel("Time [s]")
        self.canvas.draw_idle()


class CombinedTracePanel(QtWidgets.QWidget):
    """9-axis combined trace panel: Acceleration / Velocity / Displacement × Z / E / N.

    Refresh contract
    ----------------
    ``"time"``
        Move time cursors only — no HDF5 I/O, no redraw of lines.
    ``"selection"`` / ``"full"`` / ``"init"``
        Reload all three demand traces for the current node and redraw.
    Any other reason (``"demand"``, ``"component"``, ``"warp"`` …)
        No-op — traces are node-specific, not demand/component-specific.
    """

    _DEMANDS = ("accel", "vel", "disp")
    _DEMAND_TITLES = {
        "accel": "Acceleration",
        "vel":   "Velocity",
        "disp":  "Displacement",
    }
    _COMP_LABELS = ("Z", "E", "N")

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self._current_node = None
        self._trace_lines: list = []    # 9 Line2D objects in axis order
        self._time_cursors: list = []   # 9 axvline objects (animated=True)
        self._bg = None                 # blitting background cache

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.title_label = QtWidgets.QLabel("No node selected")
        self.figure = Figure(figsize=(5, 13), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(680)

        # 9 axes sharing the x-axis (time).
        self.axes = self.figure.subplots(9, 1, sharex=True)

        self._init_axes_labels()

        # draw_event fires after every full canvas.draw() — use it to cache
        # the background (without animated cursors) for cheap blitting.
        self.canvas.mpl_connect("draw_event", self._on_draw)

        layout.addWidget(self.title_label)
        layout.addWidget(self.canvas, 1)

        self.refresh("init")

    # ── Axes labelling ────────────────────────────────────────────────────────

    def _init_axes_labels(self):
        for grp, demand in enumerate(self._DEMANDS):
            for ci, comp in enumerate(self._COMP_LABELS):
                ax = self.axes[grp * 3 + ci]
                ax.set_ylabel(comp, fontsize=8)
                ax.grid(True, alpha=0.25)
                ax.tick_params(labelsize=7)
                if ci == 0:
                    ax.set_title(
                        self._DEMAND_TITLES[demand],
                        fontsize=9, fontweight="bold", loc="left", pad=2,
                    )
        self.axes[-1].set_xlabel("Time [s]", fontsize=8)

    # ── Blitting ──────────────────────────────────────────────────────────────

    def _on_draw(self, event):
        """Cache the static background after every full canvas.draw().

        Because cursors are created with ``animated=True``, matplotlib skips
        them during a regular draw, so the cached bitmap is cursor-free.  We
        then immediately redraw the cursors at their current position so the
        display looks correct after a resize or zoom.
        """
        if not self._time_cursors:
            return
        try:
            self._bg = self.canvas.copy_from_bbox(self.figure.bbox)
            t = self.session.current_time()
            for ax, cur in zip(self.axes, self._time_cursors):
                cur.set_xdata([t, t])
                ax.draw_artist(cur)
            self.canvas.blit(self.figure.bbox)
        except Exception:
            self._bg = None

    def _blit_cursors(self, t: float) -> bool:
        """Move all 9 cursors to *t* using blitting.  Returns True on success."""
        if self._bg is None or not self._time_cursors:
            return False
        try:
            self.canvas.restore_region(self._bg)
            for ax, cur in zip(self.axes, self._time_cursors):
                cur.set_xdata([t, t])
                ax.draw_artist(cur)
            self.canvas.blit(self.figure.bbox)
            return True
        except Exception:
            self._bg = None
            return False

    # ── Refresh ───────────────────────────────────────────────────────────────

    def refresh(self, reason: str = "full"):
        # ── Time: blit-only cursor update (< 1 ms, zero I/O) ─────────────────
        if reason == "time":
            if self._current_node is not None and self._time_cursors:
                t = self.session.current_time()
                if not self._blit_cursors(t):
                    # No cached background yet — fall back to draw_idle once.
                    for cur in self._time_cursors:
                        cur.set_xdata([t, t])
                    self.canvas.draw_idle()
            return

        # ── Traces don't depend on demand / component / warp / visibility ────
        if reason not in {"selection", "full", "init"}:
            return

        node_id = self.session.state.selected_node

        if node_id is None:
            self._current_node = None
            self._trace_lines = []
            self._time_cursors = []
            self._bg = None
            for ax in self.axes:
                ax.clear()
                ax.grid(True, alpha=0.25)
            self._init_axes_labels()
            self.title_label.setText("No node selected")
            self.canvas.draw_idle()
            return

        # Same node and not a forced full redraw — just sync the cursor.
        if self._current_node == node_id and reason not in {"full", "init"}:
            if self._time_cursors:
                t = self.session.current_time()
                if not self._blit_cursors(t):
                    for cur in self._time_cursors:
                        cur.set_xdata([t, t])
                    self.canvas.draw_idle()
            return

        self._current_node = node_id
        self._draw_all_traces(node_id)

    def _draw_all_traces(self, node_id):
        self._trace_lines = []
        self._time_cursors = []

        t = self.session.adapter.time
        current_t = self.session.current_time()

        try:
            traces = {
                d: self.session.adapter.trace(node_id, d)
                for d in self._DEMANDS
            }
        except Exception as exc:
            self.title_label.setText(f"Error loading traces: {exc}")
            return

        for grp, demand in enumerate(self._DEMANDS):
            data = traces[demand]   # shape (3, N_time): rows = [Z, E, N]
            for ci, comp in enumerate(self._COMP_LABELS):
                ax = self.axes[grp * 3 + ci]
                ax.clear()

                color = COMPONENT_COLORS[comp]
                (line,) = ax.plot(t, data[ci], linewidth=1.0, color=color, label=comp)
                # animated=True → matplotlib skips this artist in regular
                # draw() calls, keeping the blitting background cursor-free.
                cursor = ax.axvline(
                    current_t, color="tab:red", alpha=0.35, linewidth=1.0,
                    animated=True,
                )

                self._trace_lines.append(line)
                self._time_cursors.append(cursor)

                ax.set_ylabel(comp, fontsize=8)
                ax.grid(True, alpha=0.25)
                ax.tick_params(labelsize=7)
                if ci == 0:
                    ax.set_title(
                        self._DEMAND_TITLES[demand],
                        fontsize=9, fontweight="bold", loc="left", pad=2,
                    )

        self.axes[-1].set_xlabel("Time [s]", fontsize=8)
        self.title_label.setText(f"Node {node_id}  —  Accel / Vel / Disp")
        # Full draw establishes the blitting background via _on_draw callback.
        self._bg = None
        self.canvas.draw()


__all__ = ["CombinedTracePanel", "TracePanel", "SpectrumPanel", "AriasIntensityPanel"]
