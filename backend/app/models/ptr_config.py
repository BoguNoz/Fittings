from dataclasses import dataclass


@dataclass
class PTRConfig:
    """Configuration for PTR Hankel Transfer Matrix model."""
    l2: float = 240e-9  # PEDOT:PSS thickness [m]

    # Transducer (Al)
    k1: float = 200.0
    l1: float = 50e-9
    alfa1: float = 8.2e-5

    # Substrate (glass)
    alfa3: float = 0.5e-7
    k3: float = 1.0  # can be fitted

    # Interfaces
    r21: float = 1.0e-8

    # PEDOT:PSS properties
    rhoc: float = 2.0e6  # J/m³·K

    # Laser
    d_pump: float = 2.42e-6
    Q: float = 1.0

    # Fitting
    weight_exponent: float = 0.8
    phase_weight: float = 1.2