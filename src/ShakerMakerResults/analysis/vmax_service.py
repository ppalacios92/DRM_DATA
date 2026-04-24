"""vmax cache helpers."""

from __future__ import annotations

import json

import h5py
import numpy as np


def compute_vmax(model):
    """Compute and cache global colour limits for surface plots."""
    data_grp = model._data_grp_for_vmax
    chunk_rows = 120
    print(f"  Computing vmax (chunk mode, {chunk_rows} rows/chunk)...")
    vmax = {}
    with h5py.File(model.filename, "r") as f:
        for dtype, pkey in [
            ("accel", "acceleration"),
            ("vel", "velocity"),
            ("disp", "displacement"),
        ]:
            path = f"{data_grp}/{pkey}"
            if path not in f:
                continue
            ds = f[path]
            n_rows = ds.shape[0]
            e_max = n_max = z_max = r_max = 0.0
            for start in range(0, n_rows, chunk_rows):
                end = min(start + chunk_rows, n_rows)
                data = ds[start:end, :]
                e_d = data[0::3, :]
                n_d = data[1::3, :]
                z_d = data[2::3, :]
                e_max = max(e_max, float(np.abs(e_d).max()))
                n_max = max(n_max, float(np.abs(n_d).max()))
                z_max = max(z_max, float(np.abs(z_d).max()))
                r_max = max(r_max, float(np.sqrt(e_d**2 + n_d**2 + z_d**2).max()))
            vmax[dtype] = {
                "e": e_max,
                "n": n_max,
                "z": z_max,
                "resultant": r_max,
            }

    model._vmax = vmax
    try:
        with open(model._vmax_cache_path, "w") as cf:
            json.dump(vmax, cf)
        print(f"  vmax cached to: {model._vmax_cache_path}")
    except Exception as e:
        print(f"  vmax cache write failed (read-only filesystem?): {e}")
    return vmax
