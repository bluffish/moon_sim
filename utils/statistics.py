import numpy as np


def compute_n95(counts_per_orbit):
    """N95: orbits needed for 95% chance of >=1 eclipse, via Poisson statistics.

    counts_per_orbit can be per-orbit counts or counts scaled to any window.
    """
    p = 1 - np.exp(-counts_per_orbit)
    with np.errstate(divide='ignore', invalid='ignore'):
        N95 = np.where(p >= 1 - 1e-12, 1,
               np.where(p <= 1e-15, np.inf,
                        np.log(0.05) / np.log1p(-p)))
    return np.maximum(N95, 1)


def scale_to_window(counts_per_orbit, planet_period, window_seconds):
    """Scale per-orbit eclipse rate to a fixed observation window (in seconds)."""
    return counts_per_orbit * (window_seconds / planet_period)
