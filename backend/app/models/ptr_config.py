from dataclasses import dataclass


@dataclass
class PTRConfig:
    """Configuration for PTR model fitting."""
    l2: float = 469e-9
    k1: float = 21.0
    l1: float = 80e-9
    alfa1: float = 8.9e-6
    alfa3: float = 6.0e-6
    r21: float = 2.8e-8