"""Structured plotting API for ShakerMakerResults.

Imported lazily so optional geo dependencies are only required when needed.
"""

from importlib import import_module

__all__ = [
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
    "create_animation_map",
    "create_animation",
    "create_animation_plane",
    "plot_models_response",
    "plot_models_newmark",
    "plot_models_gf",
    "plot_models_tensor_gf",
    "plot_models_domain",
    "plot_models_arias",
]

_EXPORTS = {
    "plot_node_response": (".single_model.node_plots", "plot_node_response"),
    "plot_node_gf": (".single_model.node_plots", "plot_node_gf"),
    "plot_node_tensor_gf": (".single_model.node_plots", "plot_node_tensor_gf"),
    "plot_node_newmark": (".single_model.node_plots", "plot_node_newmark"),
    "plot_node_arias": (".single_model.node_plots", "plot_node_arias"),
    "plot_domain": (".single_model.domain_plots", "plot_domain"),
    "plot_domain_calculated_t0": (".single_model.domain_plots", "plot_domain_calculated_t0"),
    "plot_gf_connections": (".single_model.domain_plots", "plot_gf_connections"),
    "plot_calculated_vs_reused": (".single_model.domain_plots", "plot_calculated_vs_reused"),
    "plot_surface": (".single_model.surface_plots", "plot_surface"),
    "plot_surface_newmark": (".single_model.surface_plots", "plot_surface_newmark"),
    "plot_surface_arias": (".single_model.surface_plots", "plot_surface_arias"),
    "plot_surface_on_map": (".single_model.map_plots", "plot_surface_on_map"),
    "create_animation_map": (".single_model.map_plots", "create_animation_map"),
    "create_animation": (".single_model.animation_plots", "create_animation"),
    "create_animation_plane": (".single_model.animation_plots", "create_animation_plane"),
    "plot_models_response": (".comparison.response_plots", "plot_models_response"),
    "plot_models_newmark": (".comparison.response_plots", "plot_models_newmark"),
    "plot_models_gf": (".comparison.gf_plots", "plot_models_gf"),
    "plot_models_tensor_gf": (".comparison.gf_plots", "plot_models_tensor_gf"),
    "plot_models_domain": (".comparison.domain_plots", "plot_models_domain"),
    "plot_models_arias": (".comparison.arias_plots", "plot_models_arias"),
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
