"""
Regenerate plots from saved .npz data files without re-running the simulation.

Usage:
    python replot.py                      # replot all .npz files in outputs/
    python replot.py outputs/foo.npz ...  # replot specific files
"""

import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from plot.style import apply as apply_style, DPI
from plot.incl import plot_eclipse_rate, plot_n95, save_combined
from plot.heatmap import plot_heatmap_2d

apply_style()


def n95_labels(window_hrs):
    if window_hrs:
        return (
            f'{window_hrs:.0f}-hour windows needed for 95% chance of >=1 eclipse',
            f'How Many {window_hrs:.0f}-Hour Observations Until an Eclipse? ($N_{{95}}$)',
        )
    return (
        'Orbits needed for 95% chance of >=1 eclipse',
        'How Many Orbits Until an Eclipse? ($N_{95}$)',
    )


def replot(npz_path):
    data       = np.load(npz_path)
    keys       = set(data.files)
    out        = npz_path.replace('.npz', '_replot.png')
    orbits     = int(data['orbits'])
    window_hrs = float(data['window_hrs']) if 'window_hrs' in keys else 0.0
    incls_deg  = data['incls_deg']

    # Backwards-compatible N95 key lookup
    N95 = (data['N95']     if 'N95'     in keys else
           data['N95_win'] if 'N95_win' in keys else
           data['N95_12h'] if 'N95_12h' in keys else None)

    print(f"Loading {npz_path}  (keys: {sorted(keys)})")

    if 'counts_2d' in keys:
        # Heatmap data
        n95_colorbar_label = (f'{window_hrs:.0f}-hour windows needed (N95)' if window_hrs
                              else 'Orbits needed (N95)')
        n95_title          = (f'$N_{{95}}$ ({window_hrs:.0f}-hour windows)' if window_hrs
                              else '$N_{95}$ (orbits)')
        plot_heatmap_2d(data['counts_2d'], N95, incls_deg, data['a_m_vals'],
                        orbits, out,
                        n95_colorbar_label=n95_colorbar_label,
                        n95_title=n95_title)

    elif 'counts_smp' in keys and N95 is not None:
        # Inclination sweep with both panels
        ylabel, title = n95_labels(window_hrs)
        save_combined(incls_deg, data['counts_smp'], N95, orbits, out,
                      n95_ylabel=ylabel, n95_title=title)

    elif N95 is not None:
        # N95 only
        ylabel, title = n95_labels(window_hrs)
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_n95(ax, incls_deg, N95, orbits, ylabel=ylabel, title=title)
        plt.tight_layout()
        plt.savefig(out, dpi=DPI)
        plt.close()

    elif 'counts_smp' in keys:
        # Eclipse rate only
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_eclipse_rate(ax, incls_deg, data['counts_smp'])
        plt.tight_layout()
        plt.savefig(out, dpi=DPI)
        plt.close()

    else:
        print(f"  Unknown data format, skipping.")
        return

    print(f"  Saved: {out}")


if __name__ == '__main__':
    paths = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob('outputs/*.npz'))
    if not paths:
        print("No .npz files found in outputs/")
        sys.exit(0)
    for p in paths:
        replot(p)
