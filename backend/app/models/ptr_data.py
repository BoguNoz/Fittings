from dataclasses import dataclass

import numpy as np

@dataclass
class PTRData:
    frequency: np.ndarray
    amplitude: np.ndarray
    phase_deg: np.ndarray
    sample_name: str = "Unknown"
