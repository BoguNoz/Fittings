from dataclasses import dataclass

@dataclass
class PTRConfig:
    l1: float = 50e-9
    k1: float = 150.0
    alfa1: float = 2.1e-5
    l2: float = 240e-9
    rhoc2: float = 1.3e6
    alfa3: float = 0.5e-7
    k3: float = 1.0
    r21: float = 1.0e-8
    d_pump: float = 0.40e-3
    Q: float = 1.0
    anisotropy: float = 1.94
    weight_exponent: float = 0
    phase_weight: float = 1