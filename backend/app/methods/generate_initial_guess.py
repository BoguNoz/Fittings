import numpy as np
from scipy.stats import qmc


def generate_initial_guess(iteration: int, n_starts: int) -> np.ndarray:
    """
    Generate an initial parameter guess for multi-start optimization of PTR data.

    Combines Latin Hypercube Sampling with physically motivated starting points
    focused on realistic values for PEDOT:PSS thin films.

    Parameters
    ----------
    iteration : int
        Current optimization iteration index.
    n_starts : int
        Total number of multi-start optimization runs.

    Returns
    -------
    np.ndarray
        Initial parameter vector in log10 space:
        [log10(k2), log10(alfa2), log10(r32), log10(k3), phi0_deg]
    """
    # Baseline - first iteration uses value close to literature
    if iteration == 0:
        return np.array([
            np.log10(0.25),      # k2 ~ 0.25 W/m·K (literature range 0.15-0.4)
            np.log10(1.4e-7),    # alfa2 ~ 1.4e-7 m²/s
            np.log10(2e-7),      # r32 moderate interface resistance
            np.log10(3.0),       # k3 (substrate)
            0.0
        ])

    # ------------------------------------------------------------------
    # Latin Hypercube Sampling - majority of starts
    # ------------------------------------------------------------------
    if iteration <= int(n_starts * 0.65):          # ~65% LHS
        sampler = qmc.LatinHypercube(d=5, seed=42 + iteration)
        sample = sampler.random(n=1)[0]

        # Much tighter, physically motivated ranges
        ranges = np.array([
            [np.log10(0.05), np.log10(1.2)],      # k2: 0.05 – 1.2 W/m·K
            [np.log10(5e-8), np.log10(5e-7)],     # alfa2: 5e-8 – 5e-7 m²/s
            [np.log10(1e-9), np.log10(1e-6)],     # r32
            [np.log10(0.5),  np.log10(20)],       # k3
            [-60,            60]                  # phi0_deg
        ])

        p = ranges[:, 0] + sample * (ranges[:, 1] - ranges[:, 0])
        return p

    # ------------------------------------------------------------------
    # Physically motivated starting points (remaining ~35%)
    # ------------------------------------------------------------------
    else:
        idx = iteration - int(n_starts * 0.65)
        strategy = idx % 9

        if strategy == 0:   # Literature center
            return np.array([np.log10(0.22), np.log10(1.5e-7), np.log10(1.5e-7), np.log10(3.0), 2.0])

        elif strategy == 1: # Slightly higher conductivity
            return np.array([np.log10(0.45), np.log10(2.0e-7), np.log10(8e-8), np.log10(3.2), -5.0])

        elif strategy == 2: # Lower conductivity
            return np.array([np.log10(0.12), np.log10(1.0e-7), np.log10(4e-7), np.log10(2.8), 8.0])

        elif strategy == 3: # Higher interface resistance
            return np.array([np.log10(0.28), np.log10(1.6e-7), np.log10(6e-7), np.log10(3.0), 0.0])

        else:               # Small random perturbations around good region
            base = np.array([
                np.log10(0.25),
                np.log10(1.4e-7),
                np.log10(2e-7),
                np.log10(3.0),
                0.0
            ])
            perturb = np.array([
                np.random.normal(0, 0.4),
                np.random.normal(0, 0.35),
                np.random.normal(0, 0.6),
                np.random.normal(0, 0.3),
                np.random.uniform(-25, 25)
            ])
            return base + perturb