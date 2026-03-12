import os
import time
import argparse
import numpy as np
from utils.orbit import make_timesteps, get_period
from kernels.orbit_numba_chunked import count_eclipse_incl_numba_cuda_chunked
from utils.statistics import compute_n95, scale_to_window
from plot.style import apply as apply_style
from plot.heatmap import plot_heatmap_2d

GALILEAN = {
    'Io':       (4.217e8,  0.04),
    'Europa':   (6.711e8,  0.47),
    'Ganymede': (1.0704e9, 0.18),
    'Callisto': (1.8827e9, 0.19),
}

parser = argparse.ArgumentParser(description='Heatmap of eclipse rate and N95 vs inclination and moon semi-major axis')
parser.add_argument('--orbits',     type=int,   default=1000)
parser.add_argument('--divisor',    type=float, default=2.0)
parser.add_argument('--incl-steps', type=int,   default=90*4+1,
                    help='Number of inclination steps from 0 to 90 deg')
parser.add_argument('--a-p',     type=float, default=1.0e11)
parser.add_argument('--M-p',     type=float, default=2.0e30)
parser.add_argument('--omega-p', type=float, default=0.0)
parser.add_argument('--w-p',     type=float, default=0.0)
parser.add_argument('--i-p',     type=float, default=0.0)
parser.add_argument('--v0-p',    type=float, default=0.0)
parser.add_argument('--r-p',     type=float, default=6.37e6)
parser.add_argument('--a-m-min',   type=float, default=1.0e9)
parser.add_argument('--a-m-max',   type=float, default=1.0e10)
parser.add_argument('--a-m-steps', type=int,   default=50)
parser.add_argument('--M-m',     type=float, default=5.97e29)
parser.add_argument('--omega-m', type=float, default=0.0)
parser.add_argument('--w-m',     type=float, default=0.0)
parser.add_argument('--v0-m',    type=float, default=0.0)
parser.add_argument('--window',  type=float, default=None,
                    help='Observation window in hours for N95 scaling. '
                         'Omit to use the full orbital period (N95 in orbits).')
parser.add_argument('--no-galilean', action='store_true',
                    help='Omit Galilean moon markers from the plot')
parser.add_argument('--name',    type=str,   default=None,
                    help='Base name for output files')
args = parser.parse_args()

orbits  = args.orbits
divisor = args.divisor
a_p, M_p = args.a_p, args.M_p
omega_p, w_p, i_p, v0_p = args.omega_p, args.w_p, args.i_p, args.v0_p
r_p = args.r_p
M_m = args.M_m
omega_m, w_m, v0_m = args.omega_m, args.w_m, args.v0_m
window_hrs = args.window  # None means full orbital period

if window_hrs:
    name_default       = f'heatmap_{int(window_hrs)}h_{orbits}'
    n95_colorbar_label = f'{window_hrs:.0f}-hour windows needed (N95)'
    n95_title          = f'$N_{{95}}$ ({window_hrs:.0f}-hour windows)'
else:
    name_default       = f'heatmap_{orbits}'
    n95_colorbar_label = 'Orbits needed (N95)'
    n95_title          = '$N_{95}$ (orbits)'

name      = args.name or name_default
data_path = f'outputs/{name}.npz'
plot_path = f'outputs/{name}.png'
os.makedirs('outputs', exist_ok=True)
apply_style()

incls    = np.deg2rad(np.linspace(0, 90, args.incl_steps))
a_m_vals = np.linspace(args.a_m_min, args.a_m_max, args.a_m_steps)

# Use the smallest a_m to set the finest timestep resolution
script_start = time.perf_counter()
ts = make_timesteps(a_p, M_p, a_m_vals[0], M_m, r_p, divisor=divisor, orbits=orbits)

counts_2d = np.zeros((args.a_m_steps, args.incl_steps))
for i, a_m in enumerate(a_m_vals):
    print(f'a_m step {i+1}/{args.a_m_steps}  a_m={a_m:.3e}', flush=True)
    _, counts_row = count_eclipse_incl_numba_cuda_chunked(
        a_p, M_p, omega_p, w_p, i_p, v0_p,
        a_m, M_m, omega_m, w_m, incls,
        v0_m, r_p, ts, chunk_size=2048)
    counts_2d[i] = counts_row
compute_elapsed = time.perf_counter() - script_start

if window_hrs:
    lambda_n95 = scale_to_window(counts_2d, get_period(a_p, M_p), window_hrs * 3600)
else:
    lambda_n95 = counts_2d

N95       = compute_n95(lambda_n95)
incls_deg = np.rad2deg(incls)

np.savez(data_path,
         incls_deg=incls_deg, a_m_vals=a_m_vals,
         counts_2d=counts_2d, N95=N95,
         orbits=np.float64(orbits), divisor=np.float64(divisor),
         window_hrs=np.float64(window_hrs or 0.0),
         a_p=np.float64(a_p), M_p=np.float64(M_p),
         M_m=np.float64(M_m), r_p=np.float64(r_p))

galilean = None if args.no_galilean else GALILEAN
plot_heatmap_2d(counts_2d, N95, incls_deg, a_m_vals, orbits, plot_path,
                galilean=galilean,
                n95_colorbar_label=n95_colorbar_label,
                n95_title=n95_title)

total_elapsed = time.perf_counter() - script_start
print(f"Data saved to:  {data_path}")
print(f"Graph saved to: {plot_path}")
print(f"Compute time: {compute_elapsed:.3f} s")
print(f"Total time:   {total_elapsed:.3f} s")
