import os
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.orbit import *
from kernels.orbit_numba_chunked import *

os.makedirs("outputs", exist_ok=True)
plt.style.use('dark_background')

# Parameters
orbits = 50
divisor = 3

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

path = 'outputs/inclination_vs_transits.png'
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(incls, counts_smp, color='#ff6b6b', lw=2, label='moon eclipse')
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

print(f"Graph saved to: {path}")
print(f"Compute time: {compute_elapsed:.3f} s")
print(f"Total time: {total_elapsed:.3f} s")
