import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.orbit import *
from kernels.orbit_numba_chunked import *

parser = argparse.ArgumentParser(description='Plot eclipse count and N95 vs inclination')

# Simulation parameters
parser.add_argument('--orbits', type=int, default=1000)
parser.add_argument('--divisor', type=float, default=2.0)
parser.add_argument('--incl-steps', type=int, default=180*8+1,
                    help='Number of inclination steps from 0 to 180 deg')

# Planet parameters
parser.add_argument('--a-p', type=float, default=1.0e11)
parser.add_argument('--M-p', type=float, default=2.0e30)
parser.add_argument('--omega-p', type=float, default=0.0)
parser.add_argument('--w-p', type=float, default=0.0)
parser.add_argument('--i-p', type=float, default=0.0)
parser.add_argument('--v0-p', type=float, default=0.0)
parser.add_argument('--r-p', type=float, default=6.37e6)

# Moon parameters
parser.add_argument('--a-m', type=float, default=3.84e9)
parser.add_argument('--M-m', type=float, default=5.97e29)
parser.add_argument('--omega-m', type=float, default=0.0)
parser.add_argument('--w-m', type=float, default=0.0)
parser.add_argument('--v0-m', type=float, default=0.0)

# Output paths
parser.add_argument('--data-path', type=str, default=None,
                    help='Path for .npz data file (default: outputs/inclination_vs_all_{orbits}.npz)')
parser.add_argument('--plot-path', type=str, default=None,
                    help='Path for .png plot file (default: outputs/inclination_vs_all_{orbits}.png)')

args = parser.parse_args()

orbits   = args.orbits
divisor  = args.divisor
a_p, M_p = args.a_p, args.M_p
omega_p, w_p, i_p, v0_p = args.omega_p, args.w_p, args.i_p, args.v0_p
r_p = args.r_p
a_m, M_m = args.a_m, args.M_m
omega_m, w_m, v0_m = args.omega_m, args.w_m, args.v0_m

data_path = args.data_path or f'outputs/inclination_vs_all_{orbits}.npz'
plot_path = args.plot_path or f'outputs/inclination_vs_all_{orbits}.png'

os.makedirs(os.path.dirname(data_path) if os.path.dirname(data_path) else 'outputs', exist_ok=True)
os.makedirs(os.path.dirname(plot_path) if os.path.dirname(plot_path) else 'outputs', exist_ok=True)

plt.style.use('dark_background')

script_start = time.perf_counter()
ts = make_timesteps(a_p, M_p, a_m, M_m, r_p, divisor=divisor, orbits=orbits)

incls = np.deg2rad(np.linspace(0, 180, args.incl_steps))

incls, counts_smp = count_eclipse_incl_numba_cuda_chunked(
    a_p, M_p, omega_p, w_p, i_p, v0_p,
    a_m, M_m, omega_m, w_m, incls,
    v0_m, r_p, ts, chunk_size=2048)

compute_elapsed = time.perf_counter() - script_start

# N95 calculation
lambda_hat = counts_smp
p_hat = 1 - np.exp(-lambda_hat)
with np.errstate(divide='ignore', invalid='ignore'):
    N95 = np.where(p_hat >= 1 - 1e-12, 1,
           np.where(p_hat <= 1e-15, np.inf,
                    np.log(0.05) / np.log1p(-p_hat)))
N95 = np.maximum(N95, 1)
incls_deg = np.rad2deg(incls)

# Save data
np.savez(data_path,
         incls_deg=incls_deg,
         counts_smp=counts_smp,
         p_hat=p_hat,
         N95=N95,
         orbits=np.float64(orbits),
         divisor=np.float64(divisor),
         a_p=np.float64(a_p), M_p=np.float64(M_p),
         a_m=np.float64(a_m), M_m=np.float64(M_m),
         r_p=np.float64(r_p))

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

ax1.plot(incls_deg, counts_smp, color='#ff6b6b', lw=2, label='moon eclipse')
ax1.set_ylabel('Eclipses per orbit', fontsize=13)
ax1.set_title('Moon Inclination vs Eclipse Count', fontsize=15)
ax1.grid(alpha=0.2)
ax1.set_yscale('log')
ax1.legend(frameon=False, fontsize=11)

valid = np.isfinite(N95) & (N95 > 0)
ax2.plot(incls_deg[valid], N95[valid], color='#ff6b6b', lw=2, label='$N_{95}$')
ax2.set_xlabel('Moon orbital inclination', fontsize=13)
ax2.set_ylabel('Orbits needed for 95% chance of >=1 eclipse', fontsize=13)
ax2.set_title('How Many Orbits Until an Eclipse? ($N_{95}$)\n'
              f'{int(orbits)} orbits', fontsize=15)
ax2.grid(alpha=0.2)
ax2.set_yscale('log')
ax2.legend(frameon=False, fontsize=11)

plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.close()

total_elapsed = time.perf_counter() - script_start

print(f"Data saved to:  {data_path}")
print(f"Graph saved to: {plot_path}")
print(f"Compute time: {compute_elapsed:.3f} s")
print(f"Total time: {total_elapsed:.3f} s")
