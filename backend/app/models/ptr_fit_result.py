from dataclasses import dataclass
import numpy as np


@dataclass
class PTRFitResult:
    """Container for PTR fitting results."""

    # Fitted parameters
    k2: float          # Thermal conductivity of ZnO film (Layer 2) [W/(m·K)]
    alfa2: float       # Thermal diffusivity of ZnO film (Layer 2) [m²/s]
    r32: float         # Thermal boundary resistance ZnO–substrate [m²·K/W]
    k3: float          # Thermal conductivity of substrate (Layer 3) [W/(m·K)]
    phi0_deg: float    # Global phase offset [degrees]

    # Fit quality
    res_norm: float    # 2 * cost from least_squares (sum of squared weighted residuals)

    # Model and experimental curves (for plotting)
    model_amp: np.ndarray
    model_phase_deg: np.ndarray
    exp_phase_deg: np.ndarray

    phase_units: str          # Units used for input phase ('deg' or 'rad')
    pfit: np.ndarray          # Optimized parameters in log10 scale + phi0
    exit_flag: int            # Status from scipy.optimize.least_squares
    frequency_vector: np.ndarray

    l2: float = 469e-9        # Thickness of main ZnO layer used in the model [m]

    sample_name: str = "TEST"        # Name of the sample
