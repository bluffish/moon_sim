"""Plots for 2-D (inclination × semi-major axis) heatmap results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from plot.style import DPI


def plot_heatmap_2d(counts_2d, N95_2d, incls_deg, a_m_vals, orbits, path,
                    galilean=None,
                    n95_colorbar_label='12-hour windows needed (N95)',
                    n95_title='$N_{95}$ (12-hour windows)'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    counts_valid = np.where(counts_2d > 0, counts_2d, np.nan)
    finite_counts = counts_valid[np.isfinite(counts_valid)]
    im1 = ax1.pcolormesh(incls_deg, a_m_vals, counts_valid,
                         norm=LogNorm(vmin=np.nanmin(finite_counts),
                                      vmax=np.nanmax(finite_counts)),
                         cmap='inferno', shading='auto')
    fig.colorbar(im1, ax=ax1, label='Eclipses per orbit')
    ax1.set_xlabel('Moon orbital inclination (deg)', fontsize=13)
    ax1.set_ylabel('Moon semi-major axis (m)', fontsize=13)
    ax1.set_title('Eclipse Rate', fontsize=15)

    N95_valid = np.where(np.isfinite(N95_2d), N95_2d, np.nan)
    finite_N95 = N95_valid[np.isfinite(N95_valid)]
    im2 = ax2.pcolormesh(incls_deg, a_m_vals, N95_valid,
                         norm=LogNorm(vmin=np.nanmin(finite_N95),
                                      vmax=np.nanmax(finite_N95)),
                         cmap='inferno', shading='auto')
    fig.colorbar(im2, ax=ax2, label=n95_colorbar_label)
    ax2.set_xlabel('Moon orbital inclination (deg)', fontsize=13)
    ax2.set_ylabel('Moon semi-major axis (m)', fontsize=13)
    ax2.set_title(f'{n95_title}\n{int(orbits)} orbits simulated', fontsize=15)

    if galilean:
        for ax in (ax1, ax2):
            for name, (a_m_moon, incl_moon) in galilean.items():
                ax.plot(incl_moon, a_m_moon, 'o', color='white', markersize=6,
                        markeredgecolor='black', markeredgewidth=0.8)
                ax.annotate(name, (incl_moon, a_m_moon), textcoords='offset points',
                            xytext=(8, 4), color='white', fontsize=10, fontweight='bold',
                            path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()
