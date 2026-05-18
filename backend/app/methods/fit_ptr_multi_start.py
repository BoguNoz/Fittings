import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm

from app.methods.generate_initial_guess import generate_initial_guess
from app.methods.ptr_residual import ptr_residual
from app.methods.simulations_ptr_hankel import simulations_ptr_hankel
from app.models.ptr_fit_result import PTRFitResult


def fit_ptr_multi_start(
        frequency_vector: np.ndarray,
        exp_amp: np.ndarray,
        exp_phase: np.ndarray,      # already unwrapped degrees!
        n_starts: int = 1,
        **phys_params
) -> PTRFitResult:
    best_result = None
    best_resnorm = np.inf
    all_results = []
    np.random.seed(42)

    # Jeśli dane są niskoczęstotliwościowe (max < 100 kHz), wymuszamy prawie stałe parametry podłoża i anizotropię
    low_freq_mode = frequency_vector.max() < 1e5

    if low_freq_mode:
        # k2: 0.08 - 0.35, anisotropy: 1.99 - 2.01, r32: 1e-10 - 1e-5, k3: 0.999 - 1.001, phi0: -45..45
        lower = [np.log10(0.08), np.log10(1.99), np.log10(1e-10), np.log10(0.999), -45]
        upper = [np.log10(0.35), np.log10(2.01), np.log10(1e-5),  np.log10(1.001),  45]
    else:
        lower = [np.log10(0.02), np.log10(1.0), np.log10(1e-10), np.log10(0.2), -180]
        upper = [np.log10(0.8),  np.log10(10.0), np.log10(1e-5), np.log10(5.0),  180]

    rhoc = phys_params.get('rhoc', 2.5e6)

    for i in tqdm(range(n_starts), desc="Multi‑start fitting"):
        if low_freq_mode:
            # Generujemy punkty startowe głównie wokół literatury, z małym rozrzutem
            base = np.array([np.log10(0.18), np.log10(2.0), np.log10(2e-7), np.log10(1.0), 0.0])
            if i == 0:
                p0 = base
            else:
                perturb = np.array([
                    np.random.normal(0, 0.1),
                    np.random.normal(0, 0.01),
                    np.random.normal(0, 0.3),
                    np.random.normal(0, 0.001),
                    np.random.uniform(-10, 10)
                ])
                p0 = base + perturb
            # Przycinamy do bounds
            p0 = np.clip(p0, lower, upper)
        else:
            p0 = generate_initial_guess(i, n_starts)

        try:
            res = least_squares(
                ptr_residual,
                p0,
                bounds=(lower, upper),
                args=(frequency_vector, exp_amp, exp_phase),
                kwargs=phys_params,
                ftol=1e-8, xtol=1e-8, max_nfev=3000, verbose=0
            )
            cost = 2 * res.cost
            all_results.append({
                'pfit': res.x,
                'resnorm': cost,
                'status': res.status,
                'nfev': res.nfev,
                'k2': 10 ** res.x[0],
                'anisotropy': 10 ** res.x[1],
                'r32': 10 ** res.x[2],
                'k3': 10 ** res.x[3],
                'phi0_deg': res.x[4],
                'alfa2': 10 ** res.x[0] / rhoc
            })
            if cost < best_resnorm:
                best_resnorm = cost
                best_result = res
        except Exception:
            continue

    if best_result is None:
        raise RuntimeError("All multi-start attempts failed.")

    pfit = best_result.x
    k2 = 10 ** pfit[0]
    anisotropy = 10 ** pfit[1]
    r32 = 10 ** pfit[2]
    k3 = 10 ** pfit[3]
    phi0_deg = pfit[4]

    alfa2 = k2 / rhoc
    k_parallel = k2 * anisotropy

    model_kwargs = {key: phys_params[key] for key in
                    ['k1', 'l1', 'l2', 'alfa1', 'alfa3', 'r21', 'd_pump', 'Q']
                    if key in phys_params}
    _, y_complex = simulations_ptr_hankel(
        frequency_vector, k2, alfa2, r32, k3,
        anisotropy=anisotropy, **model_kwargs
    )
    y_norm = (y_complex / y_complex[0]) * np.exp(1j * np.deg2rad(phi0_deg))
    model_phase_deg = np.angle(y_norm, deg=True)

    gain = np.mean(exp_amp[:5] / np.abs(y_norm[:5]))
    model_amp_scaled = np.abs(y_norm) * gain

    exp_complex = exp_amp * np.exp(1j * np.deg2rad(exp_phase))
    e_norm = (exp_complex / exp_complex[0]) * np.exp(1j * np.deg2rad(phi0_deg))
    exp_phase_plot = np.angle(e_norm, deg=True)

    result = PTRFitResult(
        k2=k2,
        alfa2=alfa2,
        r32=r32,
        k3=k3,
        phi0_deg=phi0_deg,
        anisotropy=anisotropy,
        k_parallel=k_parallel,
        res_norm=best_resnorm,
        model_amp=model_amp_scaled,
        exp_amp=exp_amp,
        model_phase_deg=model_phase_deg,
        exp_phase_deg=exp_phase_plot,
        pfit=pfit,
        exit_flag=int(best_result.status),
        frequency_vector=frequency_vector,
        l2=phys_params.get('l2', 240e-9),
        n_starts=n_starts,
        best_resnorm=best_resnorm,
        all_results=all_results,
    )
    return result