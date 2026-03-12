#!/usr/bin/env python
"""
Fast parameter sensitivity experiment.
Runs the eclipse simulation at reduced resolution, varying one parameter at a time.
Compares each curve to the baseline to identify which parameters matter.
"""
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.orbit import make_timesteps, get_period
from kernels.orbit_numba_chunked import count_eclipse_incl_numba_cuda_chunked

# Reduced resolution for speed
ORBITS = 200
DIVISOR = 2.0
INCL_STEPS = 361  # 0.5 deg resolution

# Baseline (Earth-like defaults from plot_all_48hr.py)
BASELINE = dict(
    a_p=1.0e11, M_p=2.0e30,
    omega_p=0.0, w_p=0.0, i_p=0.0, v0_p=0.0,
    r_p=6.37e6,
    a_m=3.84e9, M_m=5.97e29,
    omega_m=0.0, w_m=0.0, v0_m=0.0,
)

# Parameters to test: name -> list of (label, value) pairs
PERTURBATIONS = {
    # --- Expected to NOT matter ---
    'v0_p':    [('π/4', np.pi/4), ('π/2', np.pi/2), ('π', np.pi)],
    'v0_m':    [('π/4', np.pi/4), ('π/2', np.pi/2), ('π', np.pi)],
    'w_p':     [('π/4', np.pi/4), ('π/2', np.pi/2), ('π', np.pi)],
    'w_m':     [('π/4', np.pi/4), ('π/2', np.pi/2), ('π', np.pi)],
    'omega_p': [('π/4', np.pi/4), ('π/2', np.pi/2), ('π', np.pi)],
    'omega_m': [('π/4', np.pi/4), ('π/2', np.pi/2), ('π', np.pi)],
    # --- Expected to MAYBE matter ---
    'i_p':     [('10°', np.deg2rad(10)), ('30°', np.deg2rad(30)), ('45°', np.deg2rad(45))],
    # --- Expected to matter (scale parameters) ---
    'a_m':     [('2×', 3.84e9*2), ('0.5×', 3.84e9*0.5), ('0.25×', 3.84e9*0.25)],
    'r_p':     [('2×', 6.37e6*2), ('0.5×', 6.37e6*0.5), ('10×', 6.37e6*10)],
    'a_p':     [('2×', 1.0e11*2), ('0.5×', 1.0e11*0.5)],
    'M_p':     [('2×', 2.0e30*2), ('0.5×', 2.0e30*0.5)],
    'M_m':     [('2×', 5.97e29*2), ('0.5×', 5.97e29*0.5)],
}


def run_one(params):
    """Run simulation with given params, return (incls_deg, counts_per_orbit)."""
    ts = make_timesteps(params['a_p'], params['M_p'],
                        params['a_m'], params['M_m'],
                        params['r_p'], divisor=DIVISOR, orbits=ORBITS)
    incls = np.deg2rad(np.linspace(0, 180, INCL_STEPS))
    incls_out, counts = count_eclipse_incl_numba_cuda_chunked(
        params['a_p'], params['M_p'], params['omega_p'], params['w_p'],
        params['i_p'], params['v0_p'],
        params['a_m'], params['M_m'], params['omega_m'], params['w_m'],
        incls, params['v0_m'], params['r_p'], ts, chunk_size=2048)
    return np.rad2deg(incls_out), counts


def main():
    os.makedirs('outputs/sensitivity', exist_ok=True)
    plt.style.use('dark_background')

    # Run baseline
    print("Running baseline...")
    t0 = time.perf_counter()
    base_incl, base_counts = run_one(BASELINE)
    print(f"  Baseline done in {time.perf_counter()-t0:.1f}s")

    # Run perturbations
    results = {}
    for param_name, variants in PERTURBATIONS.items():
        results[param_name] = []
        for label, value in variants:
            p = BASELINE.copy()
            p[param_name] = value
            print(f"Running {param_name}={label}...", end=' ', flush=True)
            t0 = time.perf_counter()
            incl, counts = run_one(p)
            elapsed = time.perf_counter() - t0
            print(f"{elapsed:.1f}s")

            # Compute max absolute and relative difference from baseline
            abs_diff = np.max(np.abs(counts - base_counts))
            rel_diff = np.max(np.abs(counts - base_counts) / np.maximum(base_counts, 1e-15))
            results[param_name].append((label, value, incl, counts, abs_diff, rel_diff))

    # --- Summary table ---
    print("\n" + "="*70)
    print(f"{'Parameter':<12} {'Variant':<8} {'Max |Δ|':<14} {'Max |Δ|/base':<14} {'Matters?'}")
    print("="*70)
    for param_name, variants in results.items():
        for label, value, incl, counts, abs_diff, rel_diff in variants:
            matters = "YES" if rel_diff > 0.05 else ("maybe" if rel_diff > 0.01 else "no")
            print(f"{param_name:<12} {label:<8} {abs_diff:<14.6f} {rel_diff:<14.6f} {matters}")
        print("-"*70)

    # --- Plot grid ---
    param_names = list(PERTURBATIONS.keys())
    n = len(param_names)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), sharex=True)
    axes = axes.flatten()

    colors = ['#4ecdc4', '#ffe66d', '#ff6b6b']
    for idx, param_name in enumerate(param_names):
        ax = axes[idx]
        ax.plot(base_incl, base_counts, color='white', lw=1.5, alpha=0.7, label='baseline')
        for j, (label, value, incl, counts, abs_diff, rel_diff) in enumerate(results[param_name]):
            ax.plot(incl, counts, color=colors[j % len(colors)], lw=1.2, label=f'{label}')
        ax.set_title(param_name, fontsize=13)
        ax.set_yscale('log')
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8, frameon=False)
        if idx >= (rows-1)*cols:
            ax.set_xlabel('Moon inclination (°)')
        if idx % cols == 0:
            ax.set_ylabel('Eclipses/orbit')

    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'Parameter Sensitivity ({ORBITS} orbits, {INCL_STEPS} incl steps)', fontsize=15, y=1.01)
    plt.tight_layout()
    outpath = 'outputs/sensitivity/param_sensitivity.png'
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {outpath}")


if __name__ == '__main__':
    main()
