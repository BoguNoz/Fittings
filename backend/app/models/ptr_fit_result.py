from dataclasses import dataclass, field
import numpy as np


@dataclass
class PTRFitResult:
    k2: float
    alfa2: float
    r32: float
    k3: float
    phi0_deg: float

    anisotropy: float = 1.0
    k_parallel: float = 0.0

    res_norm: float = 0.0
    best_resnorm: float = 0.0
    model_amp: np.ndarray = field(default_factory=lambda: np.array([]))
    exp_amp: np.ndarray = field(default_factory=lambda: np.array([]))
    model_phase_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    exp_phase_deg: np.ndarray = field(default_factory=lambda: np.array([]))

    pfit: np.ndarray = field(default_factory=lambda: np.array([]))
    frequency_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    all_results: list = field(default_factory=list)
    n_starts: int = 1