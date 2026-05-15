from dataclasses import dataclass

@dataclass
class PTRConfig:
    """Configuration for PTR model fitting – realistic values for glass/Al/PEDOT:PSS"""
    l2: float = 240e-9          # PEDOT:PSS thickness (240 nm typical)
    k1: float = 200.0           # Al thermal conductivity [W/m·K]
    l1: float = 50e-9           # Al thickness [m]
    alfa1: float = 8.2e-5       # Al thermal diffusivity [m²/s]
    alfa3: float = 0.5e-6       # Glass diffusivity (corrected)
    r21: float = 1.0e-8         # Al/PEDOT interface resistance [m²K/W]
    weight_exponent: float = 0.5   # frequency weighting for residual
    phase_weight: float = 1.5      # extra weight on imaginary part (phase)
    k3_fixed: bool = False      # if True, k3 is not fitted (use fixed value)
    fixed_k3: float = 1.0       # glass thermal conductivity [W/m·K]