from dataclasses import dataclass

@dataclass
class PTRConfig:
    """Configuration for PTR fitting with Hankel model."""
    # Sample geometry
    l2: float = 240e-9                 # PEDOT:PSS thickness

    # Transducer (Al)
    k1: float = 200.0
    l1: float = 50e-9
    alfa1: float = 8.2e-5

    # Substrate (glass)
    alfa3: float = 0.5e-6

    # Interfaces
    r21: float = 1.0e-8                # Al / PEDOT:PSS

    # Volumetric heat capacity (J/m³·K) – links α = k/ρc
    rhoc: float = 2.5e6

    # Laser spot size
    d_pump: float = 2.42e-6            # 1/e² radius (m)

    # Normalized pump power
    Q: float = 1.0

    # Fitting weights
    weight_exponent: float = 0.5
    phase_weight: float = 1.5