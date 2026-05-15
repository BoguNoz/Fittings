import numpy as np
from scipy.stats import qmc

def generate_initial_guess(iteration: int, n_starts: int) -> np.ndarray:
    """
    Generate initial guess for multi-start PTR fitting.
    Combines Latin Hypercube Sampling (70% of runs) with physically motivated points.
    Returns [log10(k2), log10(alfa2), log10(r32), log10(k3), phi0_deg].
    """
    # --- Baseline guess (first iteration) ---
    if iteration == 0:
        return np.array([
            np.log10(0.22),       # k2 ~0.22 W/m·K (PEDOT:PSS)
            np.log10(1.4e-7),     # alfa2 ~1.4e-7 m²/s
            np.log10(2e-7),       # r32 moderate
            np.log10(1.0),        # k3 = 1.0 W/m·K (glass)
            0.0
        ])

    # --- Latin Hypercube Sampling (first 70% of runs) ---
    if iteration <= int(0.7 * n_starts):
        sampler = qmc.LatinHypercube(d=5, seed=42 + iteration)
        sample = sampler.random(n=1)[0]

        # Physically plausible ranges (much wider than before)
        ranges = np.array([
            [np.log10(0.05), np.log10(1.5)],       # k2: 0.05 – 1.5 W/m·K
            [np.log10(5e-9), np.log10(5e-6)],     # alfa2: 5e-9 – 5e-6 m²/s (wide)
            [np.log10(1e-10), np.log10(1e-6)],    # r32: 1e-10 – 1e-6 m²K/W
            [np.log10(0.5), np.log10(5.0)],       # k3: 0.5 – 5.0 W/m·K
            [-180, 180]                           # phi0: full range
        ])

        p = ranges[:, 0] + sample * (ranges[:, 1] - ranges[:, 0])
        return p

    # --- Physically motivated starting points (remaining 30%) ---
    idx = iteration - int(0.7 * n_starts)
    strategy = idx % 8   # 8 different strategies

    # Literature-based values for PEDOT:PSS on glass
    lit_k2 = 0.22
    lit_alfa2 = 1.4e-7
    lit_r32 = 2e-7
    lit_k3 = 1.0

    if strategy == 0:          # Central literature
        return np.array([np.log10(lit_k2), np.log10(lit_alfa2), np.log10(lit_r32), np.log10(lit_k3), 0.0])
    elif strategy == 1:        # Higher k2, lower r32
        return np.array([np.log10(0.45), np.log10(1.6e-7), np.log10(8e-8), np.log10(1.2), 5.0])
    elif strategy == 2:        # Lower k2, higher r32
        return np.array([np.log10(0.12), np.log10(1.2e-7), np.log10(5e-7), np.log10(0.8), -10.0])
    elif strategy == 3:        # Higher diffusivity
        return np.array([np.log10(0.25), np.log10(2.5e-7), np.log10(1.5e-7), np.log10(1.0), 2.0])
    elif strategy == 4:        # Lower diffusivity
        return np.array([np.log10(0.20), np.log10(8e-8), np.log10(2e-7), np.log10(1.0), -5.0])
    else:                      # Small random perturbations around literature
        base = np.array([np.log10(lit_k2), np.log10(lit_alfa2), np.log10(lit_r32), np.log10(lit_k3), 0.0])
        perturb = np.array([
            np.random.normal(0, 0.3),
            np.random.normal(0, 0.4),
            np.random.normal(0, 0.5),
            np.random.normal(0, 0.2),
            np.random.uniform(-30, 30)
        ])
        return base + perturb