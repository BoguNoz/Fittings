import numpy as np
from scipy.stats import qmc

def generate_initial_guess(iteration: int, n_starts: int, low_freq_mode: bool = False) -> np.ndarray:
    """
    Generuje startowe 4 parametry: [log10(k2), log10(anisotropy), log10(r32), log10(k3)]
    """
    lit_k2 = 0.18
    lit_aniso = 2.0
    lit_r32 = 2e-7
    lit_k3 = 1.0

    if iteration == 0:
        return np.array([np.log10(lit_k2), np.log10(lit_aniso), np.log10(lit_r32), np.log10(lit_k3)])

    if low_freq_mode:
        # Dla niskich częstotliwości: mały rozrzut wokół literatury
        perturb = np.array([
            np.random.normal(0, 0.1),
            np.random.normal(0, 0.1),
            np.random.normal(0, 0.5),
            np.random.normal(0, 0.05)
        ])
        base = np.array([np.log10(lit_k2), np.log10(lit_aniso), np.log10(lit_r32), np.log10(lit_k3)])
        return np.clip(base + perturb,
                       [np.log10(0.05), np.log10(1.0), np.log10(1e-10), np.log10(0.8)],
                       [np.log10(1.5), np.log10(10.0), np.log10(1e-5), np.log10(1.5)])

    # Standardowa procedura – Latin Hypercube dla pierwszych 70% startów
    if iteration <= int(0.7 * n_starts):
        sampler = qmc.LatinHypercube(d=4, seed=42 + iteration)
        sample = sampler.random(n=1)[0]
        ranges = np.array([
            [np.log10(0.05), np.log10(1.5)],   # k2
            [np.log10(1.0),  np.log10(8.0)],   # anisotropy
            [np.log10(1e-10), np.log10(1e-5)], # r32
            [np.log10(0.5),  np.log10(3.0)]    # k3
        ])
        return ranges[:, 0] + sample * (ranges[:, 1] - ranges[:, 0])
    else:
        # Pozostałe starty – punkty fizycznie uzasadnione
        idx = iteration - int(0.7 * n_starts)
        strategy = idx % 5
        if strategy == 0:
            return np.array([np.log10(0.25), np.log10(2.5), np.log10(5e-8), np.log10(1.2)])
        elif strategy == 1:
            return np.array([np.log10(0.12), np.log10(1.5), np.log10(5e-7), np.log10(0.9)])
        elif strategy == 2:
            return np.array([np.log10(0.30), np.log10(3.5), np.log10(1e-7), np.log10(1.0)])
        elif strategy == 3:
            return np.array([np.log10(0.10), np.log10(1.0), np.log10(2e-7), np.log10(1.3)])
        else:
            base = np.array([np.log10(lit_k2), np.log10(lit_aniso), np.log10(lit_r32), np.log10(lit_k3)])
            perturb = np.array([
                np.random.normal(0, 0.2),
                np.random.normal(0, 0.3),
                np.random.normal(0, 0.5),
                np.random.normal(0, 0.15)
            ])
            return base + perturb