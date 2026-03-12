import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from utils.orbit import *
from kernels.orbit_numba_chunked import *

parser = argparse.ArgumentParser(description='Heatmap of eclipse count and N95 vs inclination and moon semi-major axis')

# Simulation parameters
parser.add_argument('--orbits', type=int, default=1000)
parser.add_argument('--divisor', type=float, default=2.0)
parser.add_argument('--incl-steps', type=int, default=90*4+1,
                    help='Number of inclination steps from 0 to 180 deg')

# Planet parameters
parser.add_argument('--a-p', type=float, default=1.0e11)
parser.add_argument('--M-p', type=float, default=2.0e30)
parser.add_argument('--omega-p', type=float, default=0.0)
parser.add_argument('--w-p', type=float, default=0.0)
parser.add_argument('--i-p', type=float, default=0.0)
parser.add_argument('--v0-p', type=float, default=0.0)
parser.add_argument('--r-p', type=float, default=6.37e6)

# Moon parameters (swept)
parser.add_argument('--a-m-min', type=float, default=1.0e9)
parser.add_argument('--a-m-max', type=float, default=1.0e10)
parser.add_argument('--a-m-steps', type=int, default=50)
parser.add_argument('--M-m', type=float, default=5.97e29)
parser.add_argument('--omega-m', type=float, default=0.0)
parser.add_argument('--w-m', type=float, default=0.0)
parser.add_argument('--v0-m', type=float, default=0.0)

# Output name
parser.add_argument('--name', type=str, default=None,
                    help='Base name for output files (default: heatmap_incl_a_{orbits})')

args = parser.parse_args()

orbits  = args.orbits
divisor = args.divisor
a_p, M_p = args.a_p, args.M_p
omega_p, w_p, i_p, v0_p = args.omega_p, args.w_p, args.i_p, args.v0_p
r_p = args.r_p
M_m = args.M_m
omega_m, w_m, v0_m = args.omega_m, args.w_m, args.v0_m

name = args.name or f'heatmap_incl_a_{orbits}'
data_path = f'outputs/{name}.npz'
plot_path = f'outputs/{name}.png'

os.makedirs('outputs', exist_ok=True)
plt.style.use('dark_background')

incls = np.deg2rad(np.linspace(0, 90, args.incl_steps))
a_m_vals = np.linspace(args.a_m_min, args.a_m_max, args.a_m_steps)

# Timestep computed from smallest a_m (finest resolution needed)
script_start = time.perf_counter()
ts = make_timesteps(a_p, M_p, a_m_vals[0], M_m, r_p, divisor=divisor, orbits=orbits)

counts_2d = np.zeros((args.a_m_steps, args.incl_steps), dtype=np.float64)

for i, a_m in enumerate(a_m_vals):
    print(f'a_m step {i+1}/{args.a_m_steps}  a_m={a_m:.3e}', flush=True)
    _, counts_row = count_eclipse_incl_numba_cuda_chunked(
        a_p, M_p, omega_p, w_p, i_p, v0_p,
        a_m, M_m, omega_m, w_m, incls,
        v0_m, r_p, ts, chunk_size=2048)
    counts_2d[i] = counts_row

compute_elapsed = time.perf_counter() - script_start

# N95 per 12-hour observing window
P_planet = get_period(a_p, M_p)
window_12h = 48.0 * 3600.0  # seconds
lambda_12h = counts_2d * (window_12h / P_planet)
p_12h = 1 - np.exp(-lambda_12h)
with np.errstate(divide='ignore', invalid='ignore'):
    N95_12h = np.where(p_12h >= 1 - 1e-12, 1,
               np.where(p_12h <= 1e-15, np.inf,
                        np.log(0.05) / np.log1p(-p_12h)))
N95_12h = np.maximum(N95_12h, 1)

incls_deg = np.rad2deg(incls)

# Galilean moon semi-major axes (m) and inclinations (deg to Jupiter's equator)
galilean = {
    'Io':       (4.217e8,  0.04),
    'Europa':   (6.711e8,  0.47),
    'Ganymede': (1.0704e9, 0.18),
    'Callisto': (1.8827e9, 0.19),
}

# Save data
np.savez(data_path,
         incls_deg=incls_deg,
         a_m_vals=a_m_vals,
         counts_2d=counts_2d,
         lambda_12h=lambda_12h,
         p_12h=p_12h,
         N95_12h=N95_12h,
         orbits=np.float64(orbits),
         divisor=np.float64(divisor),
         a_p=np.float64(a_p), M_p=np.float64(M_p),
         M_m=np.float64(M_m), r_p=np.float64(r_p))

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Eclipses per orbit
counts_plot = np.where(counts_2d > 0, counts_2d, np.nan)
im1 = ax1.pcolormesh(incls_deg, a_m_vals, counts_plot,
                     norm=LogNorm(vmin=np.nanmin(counts_plot[np.isfinite(counts_plot)]),
                                  vmax=np.nanmax(counts_plot)),
                     cmap='inferno', shading='auto')
fig.colorbar(im1, ax=ax1, label='Eclipses per orbit')
ax1.set_xlabel('Moon orbital inclination (deg)', fontsize=13)
ax1.set_ylabel('Moon semi-major axis (m)', fontsize=13)
ax1.set_title('Eclipse Rate', fontsize=15)

# N95
N95_plot = np.where(np.isfinite(N95_12h), N95_12h, np.nan)
im2 = ax2.pcolormesh(incls_deg, a_m_vals, N95_plot,
                     norm=LogNorm(vmin=np.nanmin(N95_plot[np.isfinite(N95_plot)]),
                                  vmax=np.nanmax(N95_plot[np.isfinite(N95_plot)])),
                     cmap='inferno', shading='auto')
fig.colorbar(im2, ax=ax2, label='12-hour windows needed (N95)')
ax2.set_xlabel('Moon orbital inclination (deg)', fontsize=13)
ax2.set_ylabel('Moon semi-major axis (m)', fontsize=13)
ax2.set_title(f'$N_{{95}}$ (12-hour windows)\n{int(orbits)} orbits simulated', fontsize=15)

# Mark Galilean moons on both heatmaps
for ax in (ax1, ax2):
    for moon_name, (a_m_moon, incl_moon) in galilean.items():
        ax.plot(incl_moon, a_m_moon, 'o', color='white', markersize=6, markeredgecolor='black', markeredgewidth=0.8)
        ax.annotate(moon_name, (incl_moon, a_m_moon), textcoords='offset points',
                    xytext=(8, 4), color='white', fontsize=10, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])

plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.close()

total_elapsed = time.perf_counter() - script_start

print(f"Data saved to:  {data_path}")
print(f"Graph saved to: {plot_path}")
print(f"Compute time: {compute_elapsed:.3f} s")
print(f"Total time: {total_elapsed:.3f} s")
