"""Visualization helpers for mode decomposition outputs.

Library layer: reusable plotting utilities called from process entrypoints.

Implementation note:
- The historical implementation lived in this module and grew large.
- It is now split into `field_plots.py` and `coeff_plots.py`.
- We re-export the public API here to preserve backward-compatible imports.
"""
from __future__ import annotations

from .diagnostics import (
    masked_weighted_r2,
    per_pixel_r2_map,
    plot_line_with_band,
    plot_scatter_true_pred,
    sample_scatter_points,
)

from .field_plots import (
    plot_domain_error_map,
    plot_domain_field_grid,
    plot_error_map,
    plot_field_grid,
    plot_lat_profile,
    plot_angular_profile,
    plot_radial_profile,
    plot_polar_field,
    plot_mesh_field,
    plot_uncertainty_map,
    plot_vector_quiver,
    plot_vector_streamplot,
)

from .coeff_plots import (
    coeff_channel_norms,
    coeff_energy_spectrum,
    coeff_energy_vector,
    coeff_value_magnitude,
    fft_magnitude_spectrum,
    plot_channel_norm_scatter,
    plot_coeff_hist_by_mode,
    plot_coeff_histogram,
    plot_coeff_spectrum,
    plot_energy_bars,
    plot_line_series,
    plot_topk_contrib,
    slepian_concentration,
    spherical_l_energy,
    wavelet_band_energy,
)

__all__ = [
    "coeff_energy_vector",
    "coeff_channel_norms",
    "coeff_value_magnitude",
    "coeff_energy_spectrum",
    "fft_magnitude_spectrum",
    "masked_weighted_r2",
    "per_pixel_r2_map",
    "plot_vector_streamplot",
    "plot_vector_quiver",
    "plot_channel_norm_scatter",
    "plot_coeff_spectrum",
    "plot_coeff_histogram",
    "plot_energy_bars",
    "plot_error_map",
    "plot_field_grid",
    "plot_line_with_band",
    "plot_line_series",
    "plot_scatter_true_pred",
    "plot_topk_contrib",
    "plot_uncertainty_map",
    "sample_scatter_points",
    "slepian_concentration",
    "spherical_l_energy",
    "wavelet_band_energy",
]
