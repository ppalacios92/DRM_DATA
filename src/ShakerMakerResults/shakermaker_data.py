"""Backward-compatible public access to :class:`ShakerMakerData`."""

from .core.shakermaker_data import DRMData, ShakerMakerData, SurfaceData

__all__ = ["ShakerMakerData", "DRMData", "SurfaceData"]
