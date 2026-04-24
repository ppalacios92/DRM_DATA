"""GF and mapping helpers for :class:`ShakerMakerData`."""

from __future__ import annotations

import h5py


def load_gf_database(model, h5_path):
    """Load GF file in OP format."""
    model._gf_h5_path = h5_path

    with h5py.File(h5_path, "r") as f:
        if "tdata" not in f:
            raise ValueError("Unsupported GF format: expected dataset 'tdata'")
        model._tdata_shape = f["tdata"].shape
        model._nt_gf = model._tdata_shape[1]
        model._t0_available = "t0" in f

    model._has_gf = True
    print(f"  GF loaded: {model._tdata_shape[0]} slots, nt={model._nt_gf}")
    print(f"  t0 available: {model._t0_available}")


def load_map(model, h5_path):
    """Load mapping file in OP format."""
    model._gf_map_h5_path = h5_path

    with h5py.File(h5_path, "r") as f:
        model._pairs_to_compute = f["pairs_to_compute"][:]
        model._pair_to_slot = f["pair_to_slot"][:]
        model._dh_of_pairs = f["dh_of_pairs"][:]
        model._zsrc_of_pairs = f["zsrc_of_pairs"][:]
        model._zrec_of_pairs = f["zrec_of_pairs"][:]
        model._delta_h = float(f["delta_h"][()])
        model._delta_v_src = float(f["delta_v_src"][()])
        model._delta_v_rec = float(f["delta_v_rec"][()])
        model._nsources = int(f["nsources"][()])
        model._nsources_db = int(f["nsources"][()])

    model._has_map = True
    model._use_pair_to_slot = True
    model._gf_loaded = True


def get_gf(model, node_id, subfault_id, component="z"):
    """Return Green's-function time series for a node/subfault pair."""
    if not model._has_gf:
        raise RuntimeError("GF not loaded. Call load_gf_database() first.")
    if not model._has_map:
        raise RuntimeError("Map not loaded. Call load_map() first.")

    node_id_num = model._n_nodes if node_id in ("QA", "qa") else node_id
    key = (node_id, subfault_id, component)

    if key not in model._gf_cache:
        slot = model._get_slot(node_id_num, subfault_id)
        comp_map = {"z": 0, "e": 1, "n": 2}

        with h5py.File(model._gf_h5_path, "r") as f:
            tdata = f["tdata"][slot]

        if component == "tdata":
            model._gf_cache[key] = tdata
        elif component in comp_map:
            model._gf_cache[key] = tdata[:, comp_map[component]]
        else:
            raise KeyError(f"Unknown component '{component}'. Use 'z','e','n','tdata'.")

    return model._gf_cache[key]
