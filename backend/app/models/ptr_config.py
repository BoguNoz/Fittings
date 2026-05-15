from dataclasses import dataclass


@dataclass
class PTRConfig:
    """Configuration for PTR model fitting."""
    l2: float = 240e-9
    k1: float = 200.0
    l1: float = 50e-9  # Warstwa Al to 50nm
    alfa1: float = 8.2e-5
    alfa2: float = -1
    alfa3: float = 0.5e-7  # POPRAWIONE: z 5.8e-7 na 0.5e-7
    r21: float = 1.0e-8
    weight: float = 3.3



