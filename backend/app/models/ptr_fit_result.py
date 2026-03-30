from dataclasses import dataclass
import numpy as np
from numpy import floating

@dataclass
class PTRFitResult:
    # Fitted physical parameters
    k2: float # Thermal conductivity of the main ZnO film (Layer 2) in W/(m·K).
    alfa2: float # Thermal diffusivity of the main ZnO film (Layer 2) in m²/s.
    r32: float # Thermal boundary resistance (TBR) between the ZnO film and the substrate (interface Layer 2–3) in m²·K/W."""
    k3: float # Thermal conductivity of the substrate (Layer 3) in W/(m·K).
    phi0_deg: float # Fitted phase offset (constant phase shift) in degrees.
    res_norm: float #Residual norm (sum of squared weighted residuals). Lower value indicates better fit."""
    model_amp: np.ndarray # Normalized amplitude of the fitted model.
    model_phase_deg: np.ndarray # Unwrapped phase of the fitted model in degrees.
    exp_phase_deg: np.ndarray #Unwrapped experimental phase converted to degrees (for direct comparison).
    phase_units: str # Units of the input experimental phase ('deg' or 'rad').
    pfit: np.ndarray # Raw optimized parameters in log10 scale: [log10(k2), log10(alfa2), log10(r32), log10(k3), phi0_deg]
    exit_flag: int # Exit status code returned by scipy.optimize.least_squares.
    frequency_vector: np.ndarray #Frequency vector used in the fitting (in Hz).
