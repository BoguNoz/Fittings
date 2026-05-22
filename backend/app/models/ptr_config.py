from dataclasses import dataclass

@dataclass
class PTRConfig:
    # Warstwy
    l1: float = 50e-9      # grubość transducera (Al)
    k1: float = 200.0      # k transducera
    alfa1: float = 8.2e-5

    l2: float = 240e-9     # PEDOT:PSS
    rhoc2: float = 2.0e6   # volumetric heat capacity

    alfa3: float = 0.5e-7  # szkło
    k3: float = 1.0

    # Interfejsy
    r21: float = 1.0e-8
    # r32 będzie fitowane

    # Laser
    d_pump: float = 2.42e-6  # 1/e² radius
    Q: float = 1.0

    # Fitting
    weight_exponent: float = 0.8
    phase_weight: float = 1.2