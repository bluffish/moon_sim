import numpy as np
from numba import cuda
import math

G = 6.67430e-11


@cuda.jit
def _count_eclipse_kernel_chunked(
    inclinations,
    xp, yp, zp,
    dp, dp2,
    cos_alpha,
    cos_m_t, sin_m_t,
    a_m, omega_m,
    chunk_size,
    out_counts,
    out_first,
    out_last
):
    idx = cuda.grid(1)
    Ni = inclinations.size
    Nt = xp.size
    n_chunks = (Nt + chunk_size - 1) // chunk_size
    if idx >= Ni * n_chunks:
        return

    # inclination as fast dimension: consecutive threads read same timestep data
    i_incl = idx % Ni
    i_chunk = idx // Ni

    inc = inclinations[i_incl]
    cos_i = math.cos(inc)
    sin_i = math.sin(inc)
    cOm = math.cos(omega_m)
    sOm = math.sin(omega_m)

    t_start = i_chunk * chunk_size
    t_end = min(t_start + chunk_size, Nt)

    prev = False
    count = 0
    first_ecl = 0

    for t_idx in range(t_start, t_end):
        cm = cos_m_t[t_idx]
        sm = sin_m_t[t_idx]

        xm_rel = a_m * (cOm * cm - sOm * sm * cos_i)
        ym_rel = a_m * (sOm * cm + cOm * sm * cos_i)
        zm_rel = a_m * (sm * sin_i)

        xm = xm_rel + xp[t_idx]
        ym = ym_rel + yp[t_idx]
        zm = zm_rel + zp[t_idx]

        dm2 = xm * xm + ym * ym + zm * zm
        cur = False

        if dm2 < dp2[t_idx]:
            dot = xp[t_idx] * xm + yp[t_idx] * ym + zp[t_idx] * zm
            dm = math.sqrt(dm2)
            denom = dp[t_idx] * dm
            if denom > 0.0:
                cos_theta = dot / denom
                cur = (cos_theta >= cos_alpha[t_idx])

        if t_idx == t_start:
            first_ecl = 1 if cur else 0

        if cur and (not prev):
            count += 1
        prev = cur

    out_counts[idx] = count
    out_first[idx] = first_ecl
    out_last[idx] = 1 if prev else 0


def count_eclipse_incl_numba_cuda_chunked(
    a_p, M_p, omega_p, w_p, i_p, v0_p,
    a_m, M_m, omega_m, w_m, inclinations,
    v0_m, r_p, t,
    dtype=np.float64,
    threads_per_block=256,
    chunk_size=8192,
):
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

    n_chunks = (Nt + chunk_size - 1) // chunk_size
    total_threads = Ni * n_chunks

    d_incl = cuda.to_device(inclinations_cpu.astype(dtype, copy=False))
    d_xp = cuda.to_device(xp)
    d_yp = cuda.to_device(yp)
    d_zp = cuda.to_device(zp)
    d_dp = cuda.to_device(dp)
    d_dp2 = cuda.to_device(dp2.astype(dtype, copy=False))
    d_cos_alpha = cuda.to_device(cos_alpha)
    d_cos_m = cuda.to_device(cos_m_t)
    d_sin_m = cuda.to_device(sin_m_t)

    d_counts = cuda.device_array(total_threads, dtype=np.float64)
    d_first = cuda.device_array(total_threads, dtype=np.int8)
    d_last = cuda.device_array(total_threads, dtype=np.int8)

    blocks = (total_threads + threads_per_block - 1) // threads_per_block
    _count_eclipse_kernel_chunked[blocks, threads_per_block](
        d_incl,
        d_xp, d_yp, d_zp,
        d_dp, d_dp2,
        d_cos_alpha,
        d_cos_m, d_sin_m,
        dtype(a_m), dtype(omega_m),
        chunk_size,
        d_counts, d_first, d_last
    )

    # Layout: idx = i_chunk * Ni + i_incl -> reshape (n_chunks, Ni), transpose
    counts = d_counts.copy_to_host().reshape(n_chunks, Ni).T   # (Ni, n_chunks)
    first = d_first.copy_to_host().reshape(n_chunks, Ni).T
    last = d_last.copy_to_host().reshape(n_chunks, Ni).T

    total = counts.sum(axis=1)

    # Boundary correction: if chunk k ends in eclipse and chunk k+1 starts in
    # eclipse, the onset counted at chunk k+1's first timestep was false
    # (prev should have been True from chunk k).
    if n_chunks > 1:
        corrections = ((last[:, :-1] == 1) & (first[:, 1:] == 1)).sum(axis=1)
        total -= corrections

    smp_counts_per_orbit = total / orbits

    return inclinations_cpu, smp_counts_per_orbit
