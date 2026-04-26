"""
ShakerMakerResults
==================
Reader and visualisation toolkit for ShakerMaker HDF5 output files.
Supports DRM outputs, SurfaceGrid outputs, and real seismic station
recordings. File format is detected automatically for HDF5 outputs.

The public API is loaded lazily so the package can be imported even when
optional viewer dependencies are not installed yet.
"""

from importlib import import_module

__all__ = [
    "ShakerMakerData",
    "DRMData",
    "SurfaceData",
    "StationData",
    "NewmarkSpectrumAnalyzer",
    "compute_vmax",
    "plot_node_response",
    "plot_node_gf",
    "plot_node_tensor_gf",
    "plot_node_newmark",
    "plot_node_arias",
    "plot_domain",
    "plot_domain_calculated_t0",
    "plot_gf_connections",
    "plot_calculated_vs_reused",
    "plot_surface",
    "plot_surface_newmark",
    "plot_surface_arias",
    "plot_surface_on_map",
    "create_animation",
    "create_animation_plane",
    "create_animation_map",
    "plot_models_response",
    "plot_models_newmark",
    "plot_models_gf",
    "plot_models_tensor_gf",
    "plot_models_domain",
    "plot_models_arias",
    "compare_node_response",
    "compare_spectra",
    "ViewerDataAdapter",
    "ViewerState",
    "ViewerSession",
]

_EXPORTS = {
    "ShakerMakerData": (".core", "ShakerMakerData"),
    "DRMData": (".core", "DRMData"),
    "SurfaceData": (".core", "SurfaceData"),
    "StationData": (".core", "StationData"),
    "NewmarkSpectrumAnalyzer": (".analysis", "NewmarkSpectrumAnalyzer"),
    "compute_vmax": (".analysis", "compute_vmax"),
    "plot_node_response": (".plotting", "plot_node_response"),
    "plot_node_gf": (".plotting", "plot_node_gf"),
    "plot_node_tensor_gf": (".plotting", "plot_node_tensor_gf"),
    "plot_node_newmark": (".plotting", "plot_node_newmark"),
    "plot_node_arias": (".plotting", "plot_node_arias"),
    "plot_domain": (".plotting", "plot_domain"),
    "plot_domain_calculated_t0": (".plotting", "plot_domain_calculated_t0"),
    "plot_gf_connections": (".plotting", "plot_gf_connections"),
    "plot_calculated_vs_reused": (".plotting", "plot_calculated_vs_reused"),
    "plot_surface": (".plotting", "plot_surface"),
    "plot_surface_newmark": (".plotting", "plot_surface_newmark"),
    "plot_surface_arias": (".plotting", "plot_surface_arias"),
    "plot_surface_on_map": (".plotting", "plot_surface_on_map"),
    "create_animation": (".plotting", "create_animation"),
    "create_animation_plane": (".plotting", "create_animation_plane"),
    "create_animation_map": (".plotting", "create_animation_map"),
    "plot_models_response": (".plotting", "plot_models_response"),
    "plot_models_newmark": (".plotting", "plot_models_newmark"),
    "plot_models_gf": (".plotting", "plot_models_gf"),
    "plot_models_tensor_gf": (".plotting", "plot_models_tensor_gf"),
    "plot_models_domain": (".plotting", "plot_models_domain"),
    "plot_models_arias": (".plotting", "plot_models_arias"),
    "compare_node_response": (".comparison", "compare_node_response"),
    "compare_spectra": (".comparison", "compare_spectra"),
    "ViewerDataAdapter": (".viewer", "ViewerDataAdapter"),
    "ViewerState": (".viewer", "ViewerState"),
    "ViewerSession": (".viewer", "ViewerSession"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

print("""
  ShakerMakerResults -- Visualization and Analysis Toolkit
  Built on top of Shakermaker Tool

  Version 1.0.0                        (c) 2026 All Rights Reserved

  Repository  :  https://github.com/ppalacios92/ShakerMakerResults
  ShakerMaker :  https://github.com/ppalacios92/ShakerMaker

  Patricio Palacios B.    |    Ladruno Team
  
  ********* (>'-')> Ladruno4ever  *********
""")
