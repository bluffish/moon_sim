import os
import time
import argparse
import numpy as np
from utils.orbit import make_timesteps, get_period
from kernels.orbit_numba_chunked import count_eclipse_incl_numba_cuda_chunked
from utils.statistics import compute_n95, scale_to_window
from plot.style import apply as apply_style
from plot.incl import save_combined

parser = argparse.ArgumentParser(description='Eclipse count and N95 vs moon orbital inclination')
parser.add_argument('--orbits',     type=int,   default=1000)
parser.add_argument('--divisor',    type=float, default=2.0)
parser.add_argument('--incl-steps', type=int,   default=180*8+1,
                    help='Number of inclination steps from 0 to 180 deg')
parser.add_argument('--a-p',     type=float, default=1.0e11)
parser.add_argument('--M-p',     type=float, default=2.0e30)
parser.add_argument('--omega-p', type=float, default=0.0)
parser.add_argument('--w-p',     type=float, default=0.0)
parser.add_argument('--i-p',     type=float, default=0.0)
parser.add_argument('--v0-p',    type=float, default=0.0)
parser.add_argument('--r-p',     type=float, default=6.37e6)
parser.add_argument('--a-m',     type=float, default=3.84e9)
parser.add_argument('--M-m',     type=float, default=5.97e29)
parser.add_argument('--omega-m', type=float, default=0.0)
parser.add_argument('--w-m',     type=float, default=0.0)
parser.add_argument('--v0-m',    type=float, default=0.0)
parser.add_argument('--window',  type=float, default=None,
                    help='Observation window in hours for N95 scaling. '
                         'Omit to use the full orbital period (N95 in orbits).')
parser.add_argument('--name',    type=str,   default=None,
                    help='Base name for output files')
args = parser.parse_args()

orbits  = args.orbits
divisor = args.divisor
a_p, M_p = args.a_p, args.M_p
omega_p, w_p, i_p, v0_p = args.omega_p, args.w_p, args.i_p, args.v0_p
r_p = args.r_p
a_m, M_m = args.a_m, args.M_m
omega_m, w_m, v0_m = args.omega_m, args.w_m, args.v0_m
window_hrs = args.window  # None means full orbital period

if window_hrs:
    name_default = f'incl_{int(window_hrs)}h_{orbits}'
    n95_ylabel   = f'{window_hrs:.0f}-hour windows needed for 95% chance of >=1 eclipse'
    n95_title    = f'How Many {window_hrs:.0f}-Hour Observations Until an Eclipse? ($N_{{95}}$)'
else:
    name_default = f'incl_{orbits}'
    n95_ylabel   = 'Orbits needed for 95% chance of >=1 eclipse'
    n95_title    = 'How Many Orbits Until an Eclipse? ($N_{95}$)'

name      = args.name or name_default
data_path = f'outputs/{name}.npz'
plot_path = f'outputs/{name}.png'
os.makedirs('outputs', exist_ok=True)
apply_style()

script_start = time.perf_counter()
ts    = make_timesteps(a_p, M_p, a_m, M_m, r_p, divisor=divisor, orbits=orbits)
incls = np.deg2rad(np.linspace(0, 180, args.incl_steps))

incls, counts_smp = count_eclipse_incl_numba_cuda_chunked(
    a_p, M_p, omega_p, w_p, i_p, v0_p,
    a_m, M_m, omega_m, w_m, incls,
    v0_m, r_p, ts, chunk_size=2048)
compute_elapsed = time.perf_counter() - script_start

if window_hrs:
    lambda_n95 = scale_to_window(counts_smp, get_period(a_p, M_p), window_hrs * 3600)
else:
    lambda_n95 = counts_smp

N95       = compute_n95(lambda_n95)
incls_deg = np.rad2deg(incls)

np.savez(data_path,
         incls_deg=incls_deg, counts_smp=counts_smp, N95=N95,
         orbits=np.float64(orbits), divisor=np.float64(divisor),
         window_hrs=np.float64(window_hrs or 0.0),
         a_p=np.float64(a_p), M_p=np.float64(M_p),
         a_m=np.float64(a_m), M_m=np.float64(M_m), r_p=np.float64(r_p))

save_combined(incls_deg, counts_smp, N95, orbits, plot_path,
              n95_ylabel=n95_ylabel, n95_title=n95_title)

total_elapsed = time.perf_counter() - script_start
print(f"Data saved to:  {data_path}")
print(f"Graph saved to: {plot_path}")
print(f"Compute time: {compute_elapsed:.3f} s")
print(f"Total time:   {total_elapsed:.3f} s")
