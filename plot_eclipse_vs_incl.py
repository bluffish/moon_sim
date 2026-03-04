import os
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.orbit import *
from kernels.orbit_numba_chunked import *

os.makedirs("outputs", exist_ok=True)
plt.style.use('dark_background')

# Parameters
orbits = 1000
divisor = 2

a_p, M_p = 1.0e11, 2.0e30
omega_p, w_p, i_p, v0_p = 0., 0., 0., 0.
r_p = 6.37e6

a_m, M_m = 3.84e9, 5.97e29
omega_m, w_m, v0_m = 0., 0., 0.

script_start = time.perf_counter()
ts = make_timesteps(a_p, M_p, a_m, M_m, r_p, divisor=divisor, orbits=orbits)

# Sweep inclination
incls = np.deg2rad(np.linspace(0, 180, 180*8+1))

counts_smp = []

incls, counts_smp = count_eclipse_incl_numba_cuda_chunked(a_p, M_p, omega_p, w_p, i_p, v0_p,
                       a_m, M_m, omega_m, w_m, incls,
                       v0_m, r_p, ts, chunk_size=2048)
compute_elapsed = time.perf_counter() - script_start
incls_deg = np.rad2deg(incls)

data_path = f'outputs/inclination_vs_transits_{orbits}.npz'
np.savez(data_path,
         incls_deg=incls_deg,
         counts_smp=counts_smp,
         orbits=np.float64(orbits),
         divisor=np.float64(divisor),
         a_p=np.float64(a_p), M_p=np.float64(M_p),
         a_m=np.float64(a_m), M_m=np.float64(M_m),
         r_p=np.float64(r_p))

path = f'outputs/inclination_vs_transits_{orbits}.png'
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(incls_deg, counts_smp, color='#ff6b6b', lw=2, label='moon eclipse')
ax.set_xlabel('Moon orbital inclination', fontsize=13)
ax.set_ylabel('Eclipses per orbit', fontsize=13)
ax.set_title('Moon Inclination vs Eclipse Count', fontsize=15)
ax.grid(alpha=0.2)
ax.set_yscale('log')
ax.legend(frameon=False, fontsize=11)
plt.tight_layout()
plt.savefig(path, dpi=150)
plt.close()
total_elapsed = time.perf_counter() - script_start

print(f"Data saved to:  {data_path}")
print(f"Graph saved to: {path}")
print(f"Compute time: {compute_elapsed:.3f} s")
print(f"Total time: {total_elapsed:.3f} s")
