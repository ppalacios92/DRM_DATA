"""Comparison plots for Arias intensity."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_models_arias"]


def plot_models_arias(
    models,
    node_ids=None,
    target_pos=None,
    data_type="accel",
    xlim=None,
    figsize=(10, 8),
    factor=1.0,
):
    """Plot Arias intensity curves for multiple models, overlaid in one figure."""
    from ..analysis.arias_intensity import AriasIntensityAnalyzer

    if node_ids is None and target_pos is None:
        raise ValueError("Provide node_ids or target_pos.")
    if len(models) != len(node_ids if node_ids else target_pos):
        raise ValueError("models and node_ids / target_pos must have the same length.")

    n = len(models)
    nids_list = node_ids if node_ids else [None] * n
    tpos_list = target_pos if target_pos else [None] * n

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    for obj, nids, tpos in zip(models, nids_list, tpos_list):
        dt = obj.time[1] - obj.time[0]

        if tpos is not None:
            nids = obj._collect_node_ids(target_pos=tpos, print_info=True)
        elif nids is not None:
            nids = obj._collect_node_ids(node_id=nids, print_info=True)

        for nid in nids:
            data, lbl = obj._resolve_node(nid, "accel")
            for ax, sig in zip(axes, (data[0], data[1], data[2])):
                ia_pct, t_start, t_end, ia_total, _ = AriasIntensityAnalyzer.compute(sig * factor / 9.81, dt)
                t = np.arange(len(ia_pct)) * dt
                (line,) = ax.plot(t, ia_pct, linewidth=1.5, label=f"{lbl} | Ia={ia_total:.3f} m/s")
                ax.axvline(t_start, color=line.get_color(), linestyle="--", linewidth=1, alpha=0.5)
                ax.axvline(t_end, color=line.get_color(), linestyle="--", linewidth=1, alpha=0.5)

    for ax, comp in zip(axes, ("Vertical (Z)", "East (E)", "North (N)")):
        ax.axhline(5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.axhline(95, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_title(f"{comp} - Arias Intensity", fontweight="bold")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("IA (%)")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        if xlim:
            ax.set_xlim(xlim)

    plt.tight_layout()
    plt.show()
