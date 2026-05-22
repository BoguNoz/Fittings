from dataclasses import dataclass
import numpy as np

@dataclass
class PTRFitResult:
    k2: float
    alfa2: float
    r32: float
    k3: float
    anisotropy: float
    k_parallel: float
    res_norm: float
    model_amp: np.ndarray
    model_phase_deg: np.ndarray
    exp_amp: np.ndarray
    exp_phase_deg: np.ndarray
    frequency_hz: np.ndarray