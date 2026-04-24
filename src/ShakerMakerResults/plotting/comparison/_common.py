"""Shared helpers for multi-model comparison plots."""

from __future__ import annotations

import h5py
import numpy as np


def _build_label(obj, node_id):
    """Build a compact display label for a model + node combination."""
    node_part = "QA" if node_id in ("QA", "qa") else f"N{node_id}"
    return f"{obj.model_name} | {node_part} | dt={obj.dt:.4f}s"


def _get_node_data(obj, node_id, data_type):
    """Return ``(z, e, n)`` signal tuple for a node, handling QA transparently."""
    if node_id in ("QA", "qa"):
        data = obj.get_qa_data(data_type)
    else:
        data = obj.get_node_data(node_id, data_type)
    return data[0], data[1], data[2]


def _get_gf_time(obj, slot):
    """Return the GF time vector for a given slot, accounting for ``t0``."""
    t0 = 0.0
    with h5py.File(obj._gf_h5_path, "r") as f:
        if "t0" in f:
            t0 = float(f["t0"][slot])
        nt = f["tdata"].shape[1]
    return np.arange(nt) * obj._dt_orig + t0
