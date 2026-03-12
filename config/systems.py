from dataclasses import dataclass


@dataclass
class SystemConfig:
    # Planet
    a_p: float
    M_p: float
    r_p: float
    omega_p: float = 0.0
    w_p:     float = 0.0
    i_p:     float = 0.0
    v0_p:    float = 0.0
    # Moon
    a_m:     float = 3.84e9
    M_m:     float = 5.97e29
    omega_m: float = 0.0
    w_m:     float = 0.0
    v0_m:    float = 0.0


EARTH_MOON = SystemConfig(
    a_p=1.0e11, M_p=2.0e30, r_p=6.37e6,
    a_m=3.84e9, M_m=5.97e29,
)

JUPITER_IO = SystemConfig(
    a_p=7.78e11, M_p=1.989e30, r_p=7.15e7,
    a_m=4.22e8,  M_m=8.93e22,
)
