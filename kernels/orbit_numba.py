import numpy as np
from numba import cuda
import math

G = 6.67430e-11


@cuda.jit
def _count_eclipse_kernel(
    inclinations,
    xp, yp, zp,
    dp, dp2,
    cos_alpha,
    cos_m_t, sin_m_t,
    a_m, omega_m,
    out_counts
):
    i = cuda.grid(1)
    Ni = inclinations.size
    if i >= Ni:
        return

    inc = inclinations[i]
    cos_i = math.cos(inc)
    sin_i = math.sin(inc)

    cOm = math.cos(omega_m)
    sOm = math.sin(omega_m)

    Nt = xp.size

    prev = False
    count = 0

    for t_idx in range(Nt):
        cm = cos_m_t[t_idx]
        sm = sin_m_t[t_idx]

        # moon relative position
        xm_rel = a_m * (cOm * cm - sOm * sm * cos_i)
        ym_rel = a_m * (sOm * cm + cOm * sm * cos_i)
        zm_rel = a_m * (sm * sin_i)

        # moon absolute position
        xm = xm_rel + xp[t_idx]
        ym = ym_rel + yp[t_idx]
        zm = zm_rel + zp[t_idx]

        dm2 = xm * xm + ym * ym + zm * zm

        # dm < dp  -> dm2 < dp2
        if dm2 >= dp2[t_idx]:
            prev = False
            continue

        # cos_theta = (p·m)/( |p||m| )
        # p·m:
        dot = xp[t_idx] * xm + yp[t_idx] * ym + zp[t_idx] * zm

        # denom = dp * dm
        dm = math.sqrt(dm2)
        denom = dp[t_idx] * dm
        if denom <= 0.0:
            prev = False
            continue

        cos_theta = dot / denom

        cur = (cos_theta >= cos_alpha[t_idx])
        if cur and (not prev):
            count += 1
        prev = cur

    out_counts[i] = count


def count_eclipse_incl_numba_cuda(
    a_p, M_p, omega_p, w_p, i_p, v0_p,
    a_m, M_m, omega_m, w_m, inclinations,
    v0_m, r_p, t,
    dtype=np.float64,
    threads_per_block=256,
):
    # NUMBA optimized eclipse counter

    inclinations_cpu = np.asarray(inclinations)
    Ni = inclinations_cpu.size

    t_cpu = np.asarray(t)
    Nt = t_cpu.size

    Pp = 2.0 * np.pi * np.sqrt(a_p**3 / (G * M_p))
    Pm = 2.0 * np.pi * np.sqrt(a_m**3 / (G * M_m))

    t_dev_in = t_cpu.astype(dtype, copy=False)

    f_p = (2.0 * np.pi * t_dev_in / Pp + v0_p).astype(dtype, copy=False)
    cos_p = np.cos(f_p + w_p).astype(dtype, copy=False)
    sin_p = np.sin(f_p + w_p).astype(dtype, copy=False)

    cOp = np.array(np.cos(omega_p), dtype=dtype)
    sOp = np.array(np.sin(omega_p), dtype=dtype)
    cIp = np.array(np.cos(i_p), dtype=dtype)
    sIp = np.array(np.sin(i_p), dtype=dtype)

    xp = a_p * (cOp * cos_p - sOp * sin_p * cIp)
    yp = a_p * (sOp * cos_p + cOp * sin_p * cIp)
    zp = a_p * (sin_p * sIp)

    dp2 = xp * xp + yp * yp + zp * zp
    dp = np.sqrt(dp2).astype(dtype, copy=False)

    ratio = (r_p / dp).astype(dtype, copy=False)
    ratio2 = ratio * ratio
    one_minus = np.maximum(dtype(0.0), dtype(1.0) - ratio2).astype(dtype, copy=False)
    cos_alpha = np.sqrt(one_minus).astype(dtype, copy=False)

    f_m = (2.0 * np.pi * t_dev_in / Pm + v0_m).astype(dtype, copy=False)
    cos_m_t = np.cos(f_m + w_m).astype(dtype, copy=False)
    sin_m_t = np.sin(f_m + w_m).astype(dtype, copy=False)

    orbits = (float(t_cpu[-1]) - float(t_cpu[0])) / float(Pp)
    if orbits < 1.0:
        orbits = 1.0

    d_incl = cuda.to_device(inclinations_cpu.astype(dtype, copy=False))
    d_xp = cuda.to_device(xp)
    d_yp = cuda.to_device(yp)
    d_zp = cuda.to_device(zp)
    d_dp = cuda.to_device(dp)
    d_dp2 = cuda.to_device(dp2.astype(dtype, copy=False))
    d_cos_alpha = cuda.to_device(cos_alpha)
    d_cos_m = cuda.to_device(cos_m_t)
    d_sin_m = cuda.to_device(sin_m_t)

    d_counts = cuda.device_array((Ni,), dtype=np.float64)

    blocks = (Ni + threads_per_block - 1) // threads_per_block
    _count_eclipse_kernel[blocks, threads_per_block](
        d_incl,
        d_xp, d_yp, d_zp,
        d_dp, d_dp2,
        d_cos_alpha,
        d_cos_m, d_sin_m,
        dtype(a_m), dtype(omega_m),
        d_counts
    )

    counts = d_counts.copy_to_host()
    smp_counts_per_orbit = counts / orbits

    return inclinations_cpu, smp_counts_per_orbit
