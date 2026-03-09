import os
import time
import numpy as np
import matplotlib.pyplot as plt
from utils.orbit import *
from kernels.orbit_numba_chunked import *

os.makedirs("outputs", exist_ok=True)
plt.style.use('dark_background')

# Parameters
orbits = 300
divisor = 2

a_p, M_p = 1.0e11, 2.0e30
omega_p, w_p, i_p, v0_p = 0., 0., 0., 0.
r_p = 6.37e6

a_m, M_m = 3.84e9, 5.97e29
omega_m, w_m, v0_m = 0., 0., 0.

script_start = time.perf_counter()
ts = make_timesteps(a_p, M_p, a_m, M_m, r_p, divisor=divisor, orbits=orbits)

# Sweep inclination
incls = np.deg2rad(np.linspace(0, 90, 90*4+1))
a = np.deg2rad(np.linspace(1e9, 1e10, 100))

for ()

# poisson model: lambda = counts / sample_orbits
# p(>=1 per orbit) = 1 - exp(-lambda)
lambda_hat = counts_smp
p_hat = 1 - np.exp(-lambda_hat)

# N_95 = ln(0.05) / ln(1 - p)
with np.errstate(divide='ignore', invalid='ignore'):
    N95 = np.where(p_hat >= 1 - 1e-12, 1,
           np.where(p_hat <= 1e-15, np.inf,
                    np.log(0.05) / np.log1p(-p_hat)))
    
N95 = np.maximum(N95, 1)
incls_deg = np.rad2deg(incls)

data_path = f'outputs/inclination_vs_N95_{orbits}.npz'
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
path = f'outputs/inclination_vs_N95_{orbits}.png'
fig, ax = plt.subplots(figsize=(10, 6))
valid = np.isfinite(N95) & (N95 > 0)
ax.plot(incls_deg[valid], N95[valid], color='#ff6b6b', lw=2, label='$N_{95}$')
ax.set_xlabel('Moon orbital inclination', fontsize=13)
ax.set_ylabel('Orbits needed for 95% chance of >=1 eclipse', fontsize=13)
ax.set_title('How Many Orbits Until an Eclipse? ($N_{95}$)\n'
             f'{int(orbits)} orbits', fontsize=15)
ax.grid(alpha=0.2)
ax.set_yscale('log')
ax.legend(frameon=False, fontsize=11)
plt.tight_layout()
plt.savefig(path, dpi=150)
plt.close()

print(f"Data saved to:  {data_path}")
print(f"Graph saved to: {path}")