"""Inline SVG icons for the Qt viewer."""

from __future__ import annotations

from functools import lru_cache

from ._imports import require_viewer_dependencies

_, _, _, QtCore, QtGui, _ = require_viewer_dependencies()


_PATHS = {
    "play": "M8 5v14l11-7z",
    "pause": "M7 5h4v14H7zM13 5h4v14h-4z",
    "step_back": "M18 6v12l-8-6zM7 6h2v12H7z",
    "step_forward": "M6 6v12l8-6zM15 6h2v12h-2z",
    "skip_back": "M19 5v14l-9-7zM8 5h2v14H8zM5 5h2v14H5z",
    "skip_forward": "M5 5v14l9-7zM14 5h2v14h-2zM17 5h2v14h-2z",
    "gear": (
        "M19.4 13.5c.1-.5.1-1 .1-1.5s0-1-.1-1.5l2-1.5-2-3.4-2.4 1"
        "c-.8-.6-1.6-1-2.6-1.3L14 2.8h-4l-.4 2.5c-.9.3-1.8.7-2.6 1.3l-2.4-1-2 3.4"
        " 2 1.5c-.1.5-.1 1-.1 1.5s0 1 .1 1.5l-2 1.5 2 3.4 2.4-1"
        "c.8.6 1.6 1 2.6 1.3l.4 2.5h4l.4-2.5c.9-.3 1.8-.7 2.6-1.3l2.4 1 2-3.4z"
        "M12 15.5A3.5 3.5 0 1 1 12 8a3.5 3.5 0 0 1 0 7.5z"
    ),
    "view_iso_ne": (
        "M12 3 20 7.5v9L12 21 4 16.5v-9L12 3zm0 2.4L7 8.2l5 2.8 5-2.8-5-2.8z"
        "M6 10v5.2l5 2.8v-5.2L6 10zm7 2.8V18l5-2.8V10l-5 2.8z"
        "M15 4h5v5h-2V7.4l-3.2 3.2-1.4-1.4L16.6 6H15V4z"
    ),
    "view_iso_nw": (
        "M12 3 20 7.5v9L12 21 4 16.5v-9L12 3zm0 2.4L7 8.2l5 2.8 5-2.8-5-2.8z"
        "M6 10v5.2l5 2.8v-5.2L6 10zm7 2.8V18l5-2.8V10l-5 2.8z"
        "M9 4H4v5h2V7.4l3.2 3.2 1.4-1.4L7.4 6H9V4z"
    ),
    "view_iso_sw": (
        "M12 3 20 7.5v9L12 21 4 16.5v-9L12 3zm0 2.4L7 8.2l5 2.8 5-2.8-5-2.8z"
        "M6 10v5.2l5 2.8v-5.2L6 10zm7 2.8V18l5-2.8V10l-5 2.8z"
        "M9 20H4v-5h2v1.6l3.2-3.2 1.4 1.4L7.4 18H9v2z"
    ),
    "view_iso_se": (
        "M12 3 20 7.5v9L12 21 4 16.5v-9L12 3zm0 2.4L7 8.2l5 2.8 5-2.8-5-2.8z"
        "M6 10v5.2l5 2.8v-5.2L6 10zm7 2.8V18l5-2.8V10l-5 2.8z"
        "M15 20h5v-5h-2v1.6l-3.2-3.2-1.4 1.4 3.2 3.2H15v2z"
    ),
    "view_top": "M4 5h16v14H4V5zm2 2v10h12V7H6zm3 3h6v4H9v-4z",
    "view_bottom": "M4 5h16v14H4V5zm2 2v10h12V7H6zm2 8h8v2H8v-2z",
    "view_front": "M5 4h14v16H5V4zm2 2v12h10V6H7zm2 3h6v6H9V9z",
    "view_back": "M5 4h14v16H5V4zm2 2v12h10V6H7zm2 2h6v2H9V8zm0 4h6v2H9v-2z",
    "view_left": "M5 5h14v14H5V5zm2 2v10h3V7H7zm5 0v10h5V7h-5z",
    "view_right": "M5 5h14v14H5V5zm2 2v10h5V7H7zm7 0v10h3V7h-3z",
    "fit": (
        "M5 10H3V3h7v2H6.4l3.3 3.3-1.4 1.4L5 6.4V10z"
        "M19 10h2V3h-7v2h3.6l-3.3 3.3 1.4 1.4L19 6.4V10z"
        "M5 14H3v7h7v-2H6.4l3.3-3.3-1.4-1.4L5 17.6V14z"
        "M19 14h2v7h-7v-2h3.6l-3.3-3.3 1.4-1.4 3.3 3.3V14z"
    ),
    "ortho": "M4 4h16v16H4V4zm2 2v5h5V6H6zm7 0v5h5V6h-5zM6 13v5h5v-5H6zm7 0v5h5v-5h-5z",
    "rotate_left_90": (
        "M8 7h7a5 5 0 1 1-4.5 7.2l1.8-.9A3 3 0 1 0 15 9H8v3L4 8l4-4v3z"
        "M5 18h4v2H5v-2zm1-5h2v4H6v-4z"
    ),
    "rotate_right_90": (
        "M16 7H9a5 5 0 1 0 4.5 7.2l-1.8-.9A3 3 0 1 1 9 9h7v3l4-4-4-4v3z"
        "M15 18h4v2h-4v-2zm1-5h2v4h-2v-4z"
    ),
}


def _svg(icon_name: str, color: str, size: int) -> bytes:
    path = _PATHS.get(icon_name)
    if path is None:
        raise KeyError(f"Unknown viewer icon '{icon_name}'.")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24"><path fill="{color}" d="{path}"/></svg>'
    ).encode("utf-8")


@lru_cache(maxsize=256)
def icon(icon_name: str, color: str = "#172033", size: int = 18) -> QtGui.QIcon:
    """Build a ``QIcon`` from the local SVG registry."""

    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.transparent)
    pixmap.loadFromData(_svg(icon_name, color, size), "SVG")
    return QtGui.QIcon(pixmap)


__all__ = ["icon"]
