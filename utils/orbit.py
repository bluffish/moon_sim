import numpy as np
# import cupy as cp
# cp.cuda.runtime.getDevice()

try:
    import cupy as cp
    cp.cuda.runtime.getDevice()
    GPU = True
except Exception:
    print("using cpu")
    GPU = False

ap = cp if GPU else np

G = 6.67430e-11 


def get_period(a, M):
    return 2 * ap.pi * ap.sqrt(a**3 / (G * M))

def _to_gpu(a):
    return cp.asarray(a) if GPU else np.asarray(a)

def _to_cpu(a):
    return cp.asnumpy(a) if GPU else np.asarray(a)


def get_pos(a, omega, i, w, M, v0, t):
    P = get_period(a, M)
    f = 2 * ap.pi * t / P + v0

    cos_fw = ap.cos(f + w)
    sin_fw = ap.sin(f + w)

    x = a * (ap.cos(omega) * cos_fw - ap.sin(omega) * sin_fw * ap.cos(i))
    y = a * (ap.sin(omega) * cos_fw + ap.cos(omega) * sin_fw * ap.cos(i))
    z = a * sin_fw * ap.sin(i)

    return x, y, z


def is_transiting(a_p, M_p, omega_p, w_p, i_p, v0_p,
                  a_m, M_m, omega_m, w_m, i_m, v0_m, r_p, t):
    t = _to_gpu(t)

    xp, yp, zp = get_pos(a_p, omega_p, i_p, w_p, M_p, v0_p, t)
    xm, ym, zm = get_pos(a_m, omega_m, i_m, w_m, M_m, v0_m, t)
    xm += xp
    ym += yp
    zm += zp

    dp = ap.sqrt(xp*xp + yp*yp + zp*zp)
    dm = ap.sqrt(xm*xm + ym*ym + zm*zm)

    cos_theta = (xp*xm + yp*ym + zp*zm) / (dp * dm)
    theta = ap.arccos(ap.clip(cos_theta, -1, 1))
    alpha = ap.arcsin(r_p / dp)

    sun_moon_planet = (dm < dp) & (theta <= alpha)
    sun_planet_moon = (dm >= dp) & (theta <= alpha)

    dxy_sq = (xm - xp)**2 + (ym - yp)**2
    planet_moon_telescope = (zm > zp) & (dxy_sq <= r_p**2)
    moon_planet_telescope = (zm <= zp) & (dxy_sq <= r_p**2)

    rx, ry, rz = xm - xp, ym - yp, zm - zp
    d_pm = ap.sqrt(rx*rx + ry*ry + rz*rz)
    moon_spot_lit = (rx/d_pm * (-xp/dp) + ry/d_pm * (-yp/dp) + rz/d_pm * (-zp/dp)) > 0
    moon_transit_on_dayside = planet_moon_telescope & moon_spot_lit

    return (_to_cpu(sun_moon_planet),
            _to_cpu(sun_planet_moon),
            _to_cpu(planet_moon_telescope),
            _to_cpu(moon_planet_telescope),
            _to_cpu(moon_transit_on_dayside),
            (_to_cpu(xm), _to_cpu(ym), _to_cpu(zm)))


def count_transits(transit_mask):
    if len(transit_mask) == 0 or not transit_mask.any():
        return 0
    d = np.diff(transit_mask.astype(np.int8))
    return int(np.sum(d == 1)) + int(transit_mask[0])


def make_timesteps(a_p, M_p, a_m, M_m, r_p, divisor=10, orbits=5):
    P = get_period(a_p, M_p)
    Pm = get_period(a_m, M_m)
    v_moon = 2 * np.pi * a_m / Pm
    dt = r_p / (divisor * v_moon)
    n_steps = int(np.ceil(P * orbits / dt))
    return np.linspace(0, P * orbits, n_steps)


def make_timesteps_rand(a_p, M_p, orbits, n_samples):
    P = get_period(a_p, M_p)
    # Pm = get_period(a_m, M_m)
    # v_moon = 2 * np.pi * a_m / Pm
    # dt = r_p / (divisor * v_moon)
    # n_steps = int(np.ceil(P * orbits / dt))
    return np.random.uniform(0, P * orbits, np.uint64(n_samples))