"""Visual theme tokens and Qt stylesheet for the interactive viewer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewerPalette:
    """Small set of UI colors used by the Qt viewer shell."""

    accent: str = "#2a6bc2"
    accent_dark: str = "#1e5299"
    navy: str = "#1e3558"
    surface: str = "#fbfaf7"
    surface_2: str = "#f2f5f7"
    surface_3: str = "#e6ebef"
    text: str = "#172033"
    text_2: str = "#4b5d74"
    text_muted: str = "#7d8a9a"
    border: str = "#d7dee6"
    border_strong: str = "#aeb9c7"
    danger: str = "#b24a3f"
    plot_white: str = "#ffffff"


LIGHT_PALETTE = ViewerPalette()


def build_stylesheet(palette: ViewerPalette = LIGHT_PALETTE) -> str:
    """Return the global Qt stylesheet for the viewer."""

    return f"""
    QMainWindow {{
        background: {palette.surface_2};
        color: {palette.text};
    }}
    QWidget#ViewerCentral {{
        background: {palette.surface_2};
        color: {palette.text};
    }}
    QWidget#ViewerHeader {{
        background: {palette.surface};
        border: 1px solid {palette.border};
        border-radius: 6px;
    }}
    QWidget#ViewerHeader QLabel {{
        color: {palette.navy};
    }}
    QWidget#TimeControls {{
        background: {palette.surface};
        border: 1px solid {palette.border};
        border-radius: 6px;
    }}
    QLabel#TimeLabel {{
        color: {palette.accent_dark};
        font-family: Consolas, "SF Mono", "Cascadia Mono", monospace;
        font-weight: 600;
        padding: 0 8px;
    }}
    QLabel#SpeedLabel {{
        color: {palette.text_muted};
        font-size: 11px;
    }}
    QLabel#StatusChip {{
        padding: 2px 9px;
        border-radius: 9px;
        background: {palette.surface};
        border: 1px solid {palette.border};
        color: {palette.text_2};
        font-family: Consolas, "SF Mono", "Cascadia Mono", monospace;
        font-size: 11px;
    }}
    QToolButton#IconButton,
    QToolButton#TextIconButton,
    QPushButton {{
        min-height: 24px;
        border: 1px solid {palette.border_strong};
        border-radius: 5px;
        background: {palette.surface};
        color: {palette.text};
        padding: 2px 8px;
    }}
    QToolButton#IconButton {{
        min-width: 26px;
        max-width: 26px;
        padding: 2px;
    }}
    QToolButton#PlayButton {{
        min-height: 28px;
        border: 1px solid {palette.accent};
        border-radius: 5px;
        background: #e8f0fe;
        color: {palette.navy};
        font-weight: 600;
        padding: 2px 12px;
    }}
    QToolButton#IconButton:hover,
    QToolButton#TextIconButton:hover,
    QPushButton:hover {{
        background: {palette.surface_3};
        border-color: {palette.accent};
    }}
    QToolButton#PlayButton:hover {{
        background: {palette.surface_3};
        border-color: {palette.accent_dark};
    }}
    QToolButton#IconButton:checked,
    QToolButton#TextIconButton:checked {{
        background: #e8f0fe;
        border-color: {palette.accent};
        color: {palette.navy};
    }}
    QSlider::groove:horizontal {{
        height: 6px;
        border-radius: 3px;
        background: {palette.surface_3};
    }}
    QSlider::sub-page:horizontal {{
        border-radius: 3px;
        background: {palette.accent};
    }}
    QSlider::handle:horizontal {{
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
        background: #ffffff;
        border: 2px solid {palette.accent};
    }}
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QLineEdit {{
        min-height: 24px;
        border: 1px solid {palette.border};
        border-radius: 5px;
        background: #ffffff;
        color: {palette.text};
        padding: 1px 6px;
    }}
    QDoubleSpinBox:focus,
    QSpinBox:focus,
    QComboBox:focus,
    QLineEdit:focus {{
        border-color: {palette.accent};
    }}
    QMenu {{
        background: {palette.surface};
        border: 1px solid {palette.border};
        color: {palette.text};
    }}
    QTabWidget::pane {{
        border: 1px solid {palette.border};
        border-radius: 5px;
        background: {palette.surface};
        top: -1px;
    }}
    QTabBar::tab {{
        min-height: 24px;
        padding: 4px 13px;
        border: 1px solid {palette.border};
        border-bottom-color: {palette.border};
        background: {palette.surface_2};
        color: {palette.text_2};
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background: {palette.surface};
        color: {palette.navy};
        border-color: {palette.accent};
        border-bottom-color: {palette.surface};
        font-weight: 600;
    }}
    QGroupBox {{
        border: 1px solid {palette.border};
        border-radius: 6px;
        margin-top: 6px;
        padding: 8px 6px 6px 6px;
        background: {palette.surface};
        color: {palette.navy};
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
        color: {palette.navy};
        background: {palette.surface};
    }}
    QWidget#CameraViewRail {{
        background: {palette.surface};
        border-right: 1px solid {palette.border};
    }}
    QLabel#ViewRailSection {{
        color: {palette.text_muted};
        font-size: 10px;
        font-weight: 600;
        padding: 4px 0 2px 0;
    }}
    QToolButton#ViewRailButton {{
        min-width: 44px;
        max-width: 44px;
        min-height: 42px;
        border: 1px solid transparent;
        border-radius: 5px;
        background: transparent;
        color: {palette.navy};
        padding: 2px;
        font-size: 10px;
    }}
    QToolButton#ViewRailButton:hover {{
        background: {palette.surface_3};
        border-color: {palette.border};
    }}
    QToolButton#ViewRailButton:checked {{
        background: #e8f0fe;
        border-color: {palette.accent};
        color: {palette.navy};
        font-weight: 600;
    }}
    QStatusBar {{
        background: {palette.surface_2};
        border-top: 1px solid {palette.border};
    }}
    """


__all__ = ["LIGHT_PALETTE", "ViewerPalette", "build_stylesheet"]
