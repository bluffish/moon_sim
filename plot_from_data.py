"""
Load saved .npz data files from outputs/ and regenerate plots.

Usage:
    python plot_from_data.py                      # replot all .npz files in outputs/
    python plot_from_data.py outputs/foo.npz ...  # replot specific files
"""

import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')


def plot_eclipse_counts(data, out_path):
    incls_deg  = data['incls_deg']
    counts_smp = data['counts_smp']
    orbits     = int(data['orbits'])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(incls_deg, counts_smp, color='#ff6b6b', lw=2, label='moon eclipse')
    ax.set_xlabel('Moon orbital inclination', fontsize=13)
    ax.set_ylabel('Eclipses per orbit', fontsize=13)
    ax.set_title('Moon Inclination vs Eclipse Count', fontsize=15)
    ax.grid(alpha=0.2)
    ax.set_yscale('log')
    ax.legend(frameon=False, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_n95(data, out_path):
    incls_deg = data['incls_deg']
    N95       = data['N95']
    orbits    = int(data['orbits'])

    fig, ax = plt.subplots(figsize=(10, 6))
    valid = np.isfinite(N95) & (N95 > 0)
    ax.plot(incls_deg[valid], N95[valid], color='#ff6b6b', lw=2, label='$N_{95}$')
    ax.set_xlabel('Moon orbital inclination', fontsize=13)
    ax.set_ylabel('Orbits needed for 95% chance of >=1 eclipse', fontsize=13)
    ax.set_title('How Many Orbits Until an Eclipse? ($N_{95}$)\n'
                 f'{orbits} orbits', fontsize=15)
    ax.grid(alpha=0.2)
    ax.set_yscale('log')
    ax.legend(frameon=False, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def replot(npz_path):
    data = np.load(npz_path)
    keys = set(data.files)
    print(f"Loading {npz_path}  (keys: {sorted(keys)})")

    orbits = int(data['orbits'])

    if 'N95' in keys:
        out = npz_path.replace('.npz', '_replot.png')
        plot_n95(data, out)
    elif 'counts_smp' in keys:
        out = npz_path.replace('.npz', '_replot.png')
        plot_eclipse_counts(data, out)
    else:
        print(f"  Unknown data format, skipping.")


if __name__ == '__main__':
    paths = sys.argv[1:] if len(sys.argv) > 1 else sorted(glob.glob('outputs/*.npz'))
    if not paths:
        print("No .npz files found in outputs/")
        sys.exit(0)
    for p in paths:
        replot(p)
