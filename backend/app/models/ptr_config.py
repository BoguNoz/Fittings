from dataclasses import dataclass

@dataclass
class PTRConfig:
    l1: float = 50e-9
    k1: float = 200.0
    alfa1: float = 8.2e-5
    l2: float = 240e-9
    rhoc2: float = 2.0e6
    alfa3: float = 0.5e-7
    k3: float = 1.0
    r21: float = 1.0e-8
    d_pump: float = 2.42e-6
    Q: float = 1.0
    weight_exponent: float = 0.8
    phase_weight: float = 1.2