import numpy as np
from scipy.stats import qmc


def generate_initial_guess(iteration: int, n_starts: int) -> np.ndarray:
    """
    Generate initial guess for multi-start fitting with Hankel model.
    Returns [log10(k2), log10(anisotropy), log10(r32), log10(k3), phi0_deg]
    """
    # Baseline (literature for PEDOT:PSS on glass)
    lit_k2 = 0.18          # cross-plane conductivity ~0.18 W/m·K
    lit_aniso = 2.0        # k∥/k⊥ ≈ 2 (in-plane ~0.36)
    lit_r32 = 2e-7         # boundary resistance ~2e-7 m²K/W
    lit_k3 = 1.0           # glass substrate
    lit_phi0 = 0.0

    if iteration == 0:
        return np.array([np.log10(lit_k2), np.log10(lit_aniso),
                         np.log10(lit_r32), np.log10(lit_k3), lit_phi0])

    # Latin Hypercube for first 70% of starts
    if iteration <= int(0.7 * n_starts):
        sampler = qmc.LatinHypercube(d=5, seed=42 + iteration)
        sample = sampler.random(n=1)[0]
        ranges = np.array([
            [np.log10(0.05), np.log10(0.6)],      # k2: 0.05 – 0.6 W/m·K
            [np.log10(1.0),  np.log10(8.0)],      # anisotropy: 1 – 8
            [np.log10(1e-10), np.log10(1e-5)],    # r32: 1e-10 – 1e-5
            [np.log10(0.5),  np.log10(3.0)],      # k3: 0.5 – 3.0 W/m·K
            [-180, 180]                           # phi0 full range
        ])
        return ranges[:, 0] + sample * (ranges[:, 1] - ranges[:, 0])

    # Physically motivated points (last 30%)
    idx = iteration - int(0.7 * n_starts)
    strategy = idx % 6
    if strategy == 0:
        return np.array([np.log10(lit_k2), np.log10(lit_aniso),
                         np.log10(lit_r32), np.log10(lit_k3), 0.0])
    elif strategy == 1:
        return np.array([np.log10(0.25), np.log10(2.5), np.log10(5e-8),
                         np.log10(1.2), 5.0])
    elif strategy == 2:
        return np.array([np.log10(0.12), np.log10(1.5), np.log10(5e-7),
                         np.log10(0.8), -10.0])
    elif strategy == 3:
        return np.array([np.log10(0.30), np.log10(3.5), np.log10(1e-7),
                         np.log10(1.0), 2.0])
    elif strategy == 4:
        return np.array([np.log10(0.10), np.log10(1.0), np.log10(2e-7),
                         np.log10(1.5), -5.0])
    else:
        base = np.array([np.log10(lit_k2), np.log10(lit_aniso),
                         np.log10(lit_r32), np.log10(lit_k3), 0.0])
        perturb = np.array([
            np.random.normal(0, 0.2),
            np.random.normal(0, 0.3),
            np.random.normal(0, 0.5),
            np.random.normal(0, 0.15),
            np.random.uniform(-30, 30)
        ])
        return base + perturb