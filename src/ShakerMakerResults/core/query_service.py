"""Data-access helpers for :class:`ShakerMakerData`."""

from __future__ import annotations

import gc

import h5py
import numpy as np
from scipy.interpolate import interp1d


def get_node_data(model, node_id, data_type="accel"):
    """Return signal array ``(3, Nt)`` for a single node."""
    key = (node_id, data_type)
    if key not in model._node_cache:
        idx = 3 * node_id
        path = {
            "accel": f"{model._data_grp}/acceleration",
            "vel": f"{model._data_grp}/velocity",
            "disp": f"{model._data_grp}/displacement",
        }[data_type]
        with h5py.File(model.filename, "r") as f:
            data = f[path][idx : idx + 3, :]
        if hasattr(model, "_window_mask"):
            data = data[:, model._window_mask]
        elif hasattr(model, "_resample_cache"):
            rs = np.zeros((3, len(model.time)))
            for i in range(3):
                rs[i] = interp1d(
                    model._resample_cache["time_orig"],
                    data[i],
                    kind="linear",
                    fill_value="extrapolate",
                )(model.time)
            data = rs
        data = data[[2, 0, 1], :]
        model._node_cache[key] = data
    return model._node_cache[key]


def get_qa_data(model, data_type="accel"):
    """Return signal array ``(3, Nt)`` for the QA station."""
    if model._qa_grp is None:
        raise AttributeError("QA station only available in DRM output files.")
    key = ("qa", data_type)
    if key not in model._node_cache:
        path = {
            "accel": f"{model._qa_grp}/acceleration",
            "vel": f"{model._qa_grp}/velocity",
            "disp": f"{model._qa_grp}/displacement",
        }[data_type]
        with h5py.File(model.filename, "r") as f:
            data = f[path][:]
        if hasattr(model, "_window_mask"):
            data = data[:, model._window_mask]
        elif hasattr(model, "_resample_cache"):
            rs = np.zeros((3, len(model.time)))
            for i in range(3):
                rs[i] = interp1d(
                    model._resample_cache["time_orig"],
                    data[i],
                    kind="linear",
                    fill_value="extrapolate",
                )(model.time)
            data = rs
        data = data[[2, 0, 1], :]
        model._node_cache[key] = data
    return model._node_cache[key]


def get_surface_snapshot(model, time_idx, component="z", data_type="vel"):
    """Return signal values for all nodes at a single time index."""
    row = {"e": 0, "n": 1, "z": 2}[component.lower()]
    path = {
        "accel": f"{model._data_grp}/acceleration",
        "vel": f"{model._data_grp}/velocity",
        "disp": f"{model._data_grp}/displacement",
    }[data_type]
    with h5py.File(model.filename, "r") as f:
        return f[path][row::3, time_idx]


def clear_cache(model):
    """Release all in-memory cached data."""
    model._node_cache.clear()
    model._gf_cache.clear()
    model._spectrum_cache.clear()
    gc.collect()
    print("Cache cleared.")
