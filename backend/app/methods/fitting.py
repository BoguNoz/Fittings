import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc
from tqdm import tqdm

from app.methods.simulations import simulations_ptr_hankel
from app.models.ptr_fit_result import PTRFitResult


def fit_ptr_multi_start(frequency_vector, exp_amp, exp_phase, n_starts=25, **phys_params):
    """Multi-start with better error handling."""
    best_resnorm = np.inf
    best_pfit = None
    best_res = None
    all_results = []

    lb = [np.log10(0.05), np.log10(1.0), np.log10(1e-9), np.log10(0.5)]
    ub = [np.log10(1.0), np.log10(6.0), np.log10(1e-5), np.log10(2.5)]

    print(f"Starting multi-start with {n_starts} attempts...")

    for i in tqdm(range(n_starts), desc="Multi-start fitting"):
        p0 = _generate_initial_guess(i, n_starts)
        try:
            res = least_squares(
                _ptr_residual,
                p0,
                bounds=(lb, ub),
                args=(frequency_vector, exp_amp, exp_phase),
                kwargs=phys_params,
                max_nfev=1000,
                ftol=1e-8,
                xtol=1e-8
            )

            cost = 2 * res.cost
            all_results.append({'pfit': res.x.copy(), 'cost': cost, 'status': res.status})

            if cost < best_resnorm and np.isfinite(cost):
                best_resnorm = cost
                best_pfit = res.x.copy()
                best_res = res

        except Exception as e:
            print(f"Attempt {i} failed with p0={p0}: {type(e).__name__} - {str(e)[:80]}")
            continue

    if best_pfit is None:
        raise RuntimeError("All multi-start attempts failed!")

    # === Best result reconstruction ===
    k2 = 10 ** best_pfit[0]
    anisotropy = 10 ** best_pfit[1]
    r32 = 10 ** best_pfit[2]
    k3 = 10 ** best_pfit[3]
    rhoc = phys_params.get('rhoc', 2.0e6)
    alfa2 = k2 / rhoc
    k_parallel = k2 * anisotropy

    _, y_complex = simulations_ptr_hankel(
        frequency_vector, k2, alfa2, r32, k3, anisotropy, **phys_params
    )

    exp_complex = exp_amp * np.exp(1j * np.unwrap(np.deg2rad(exp_phase)))
    G = np.vdot(y_complex, exp_complex) / (np.vdot(y_complex, y_complex) + 1e-30)
    model_scaled = G * y_complex

    return PTRFitResult(
        k2=k2,
        alfa2=alfa2,
        r32=r32,
        k3=k3,
        phi0_deg=np.angle(G, deg=True),
        anisotropy=anisotropy,
        k_parallel=k_parallel,
        res_norm=best_resnorm,
        best_resnorm=best_resnorm,
        model_amp=np.abs(model_scaled),
        exp_amp=exp_amp,
        model_phase_deg=np.angle(model_scaled, deg=True),
        exp_phase_deg=exp_phase,
        pfit=best_pfit,
        frequency_vector=frequency_vector,
        all_results=all_results,
        n_starts=n_starts
    )


def _generate_initial_guess(i, n_starts):
    if i == 0:
        return np.array([np.log10(0.25), np.log10(2.0), np.log10(2e-7), np.log10(1.0)])
    sampler = qmc.LatinHypercube(d=4, seed=42 + i)
    sample = sampler.random(1)[0]
    ranges = np.array([
        [np.log10(0.08), np.log10(0.6)],
        [np.log10(1.0), np.log10(5.0)],
        [np.log10(5e-9), np.log10(8e-6)],
        [np.log10(0.6), np.log10(1.8)]
    ])
    return ranges[:, 0] + sample * (ranges[:, 1] - ranges[:, 0])


def _ptr_residual(p, freq, exp_amp, exp_phase, **params):
    """MUST return 1D array!"""
    try:
        k2 = 10 ** p[0]
        aniso = 10 ** p[1]
        r32 = 10 ** p[2]
        k3 = 10 ** p[3]
        alfa2 = k2 / params.get('rhoc', 2.0e6)

        _, model_complex = simulations_ptr_hankel(freq, k2, alfa2, r32, k3, aniso, **params)

        exp_complex = exp_amp * np.exp(1j * np.unwrap(np.deg2rad(exp_phase)))
        G = np.vdot(model_complex, exp_complex) / (np.vdot(model_complex, model_complex) + 1e-30)
        model_scaled = G * model_complex

        # Weighting
        w = (freq / freq.max() + 1e-8) ** params.get('weight_exponent', 0.7)

        diff = (model_scaled - exp_complex) / np.maximum(np.abs(exp_complex), 1e-12)
        diff = diff * w[:, np.newaxis]  # ← tu był błąd wymiarów

        # Flatten to 1D - CRITICAL for least_squares
        residuals = np.concatenate([
            diff.real.ravel(),
            params.get('phase_weight', 1.2) * diff.imag.ravel()
        ])

        return residuals

    except Exception as e:
        raise ValueError(f"Residual computation failed: {str(e)[:120]}") from e