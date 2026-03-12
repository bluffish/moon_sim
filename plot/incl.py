"""Plots for inclination-sweep results."""

import numpy as np
import matplotlib.pyplot as plt
from plot.style import COLOR, DPI, GRID_ALPHA


def plot_eclipse_rate(ax, incls_deg, counts_per_orbit):
    ax.plot(incls_deg, counts_per_orbit, color=COLOR, lw=2, label='moon eclipse')
    ax.set_ylabel('Eclipses per orbit', fontsize=13)
    ax.set_title('Moon Inclination vs Eclipse Count', fontsize=15)
    ax.grid(alpha=GRID_ALPHA)
    ax.set_yscale('log')
    ax.legend(frameon=False, fontsize=11)


def plot_n95(ax, incls_deg, N95, orbits,
             ylabel='Orbits needed for 95% chance of >=1 eclipse',
             title='How Many Orbits Until an Eclipse? ($N_{95}$)'):
    valid = np.isfinite(N95) & (N95 > 0)
    ax.plot(incls_deg[valid], N95[valid], color=COLOR, lw=2, label='$N_{95}$')
    ax.set_xlabel('Moon orbital inclination', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(f'{title}\n{int(orbits)} orbits simulated', fontsize=15)
    ax.grid(alpha=GRID_ALPHA)
    ax.set_yscale('log')
    ax.legend(frameon=False, fontsize=11)


def save_combined(incls_deg, counts_per_orbit, N95, orbits, path,
                  n95_ylabel='Orbits needed for 95% chance of >=1 eclipse',
                  n95_title='How Many Orbits Until an Eclipse? ($N_{95}$)'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    plot_eclipse_rate(ax1, incls_deg, counts_per_orbit)
    plot_n95(ax2, incls_deg, N95, orbits, ylabel=n95_ylabel, title=n95_title)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()
