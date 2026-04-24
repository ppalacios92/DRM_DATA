"""Windowing helpers for :class:`ShakerMakerData`."""

from __future__ import annotations

import numpy as np


def get_window(model, t_start, t_end):
    """Return a time-windowed copy without reading signal data eagerly."""
    new = model.__class__.__new__(model.__class__)
    for a, v in model.__dict__.items():
        setattr(new, a, v)
    mask = (model.time >= t_start) & (model.time <= t_end)
    new._window_mask = mask
    new._n_time_data = int(mask.sum())
    new.time = model.time[mask]
    new.name = f"{model.name} [{t_start}-{t_end}s]"
    new._node_cache = {}
    new._gf_cache = {}
    new._spectrum_cache = {}
    new._vmax = model._vmax
    print(f"Window [{t_start}, {t_end}]s -> {new._n_time_data} samples")
    return new


def resample(model, dt):
    """Return a lazily resampled copy of the model."""
    new = model.__class__.__new__(model.__class__)
    for a, v in model.__dict__.items():
        setattr(new, a, v)
    t_orig = np.arange(model._n_time_data) * model._dt_orig + model._tstart
    gf_orig = np.arange(model._n_time_gf) * model._dt_orig
    new.dt = dt
    new.time = np.arange(t_orig[0], t_orig[-1], dt)
    new.gf_time = (
        np.arange(gf_orig[0], gf_orig[-1], dt)
        if len(gf_orig) > 0
        else np.array([])
    )
    new._resample_cache = {"time_orig": t_orig, "gf_time_orig": gf_orig}
    new._node_cache = {}
    new._gf_cache = {}
    new._spectrum_cache = {}
    new._vmax = model._vmax

    sep = "--" * 50
    print(sep)
    print("Resample")
    print(f"  dt       :  {model._dt_orig}s  ->  {dt}s")
    print(f"  Steps    :  {model._n_time_data}  ->  {len(new.time)}")
    print(f"  Duration :  {t_orig[0]:.3f}s  -  {t_orig[-1]:.3f}s  (unchanged)")
    print(sep + "\n")
    return new
