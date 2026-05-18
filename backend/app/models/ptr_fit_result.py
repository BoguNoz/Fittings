from dataclasses import dataclass, field
import numpy as np

@dataclass
class PTRFitResult:
    """Container for PTR fitting results (Hankel model)."""
    k2: float                # cross-plane thermal conductivity (W/m·K)
    alfa2: float             # cross-plane thermal diffusivity (m²/s)
    r32: float               # film/substrate boundary resistance (m²·K/W)
    k3: float                # substrate conductivity (W/m·K)
    phi0_deg: float          # global phase offset (deg)

    anisotropy: float = 1.0  # k_parallel / k_perp
    k_parallel: float = 0.0  # in-plane conductivity (W/m·K)

    res_norm: float = 0.0    # 2*cost
    model_amp: np.ndarray = field(default_factory=lambda: np.array([]))
    exp_amp: np.ndarray = field(default_factory=lambda: np.array([]))
    model_phase_deg: np.ndarray = field(default_factory=lambda: np.array([]))
    exp_phase_deg: np.ndarray = field(default_factory=lambda: np.array([]))

    pfit: np.ndarray = field(default_factory=lambda: np.array([]))
    exit_flag: int = 0
    frequency_vector: np.ndarray = field(default_factory=lambda: np.array([]))

    l2: float = 240e-9
    n_starts: int = 25
    best_resnorm: float = 0.0
    all_results: list = field(default_factory=list)