import numpy as np
from scipy.optimize import least_squares

from app.methods.generate_initial_guess import generate_initial_guess
from app.methods.ptr_residual import ptr_residual
from app.methods.simulations_ptr import simulations_ptr
from app.models.ptr_fit_result import PTRFitResult


def fit_ptr_multi_start(
        frequency_vector: np.ndarray,
        exp_amp: np.ndarray,
        exp_phase: np.ndarray,
        n_starts: int = 25,
        **phys_params
) -> PTRFitResult:
    """
    Multi-start version of PTR fitting.
    Runs optimization from many different initial points and returns the best one.
    """
    best_result = None
    best_resnorm = np.inf
    all_results = []

    np.random.seed(42)

    for i in range(n_starts):
        p0 = generate_initial_guess(i, n_starts)

        try:
            res = least_squares(
                ptr_residual,
                p0,
                bounds=([np.log10(1e-3), np.log10(1e-10), np.log10(1e-10), np.log10(0.1), -360],
                        [np.log10(500), np.log10(1e-3), np.log10(1e-4), np.log10(100), 360]),
                args=(frequency_vector, exp_amp, exp_phase),
                kwargs=phys_params,
                ftol=1e-8,
                xtol=1e-8,
                max_nfev=2000,
                verbose=0
            )

            current_resnorm = 2 * res.cost
            print(current_resnorm)

            all_results.append({
                'pfit': res.x,
                'resnorm': current_resnorm,
                'status': res.status,
                'nfev': res.nfev
            })

            if current_resnorm < best_resnorm:
                best_resnorm = current_resnorm
                best_result = res
                best_pfit = res.x

        except Exception as e:
            continue

    if best_result is None:
        raise RuntimeError("All multi-start attempts failed.")

    pfit = best_pfit
    k2 = 10 ** pfit[0]
    alfa2 = 10 ** pfit[1]
    r32 = 10 ** pfit[2]
    k3 = 10 ** pfit[3]
    phi0_deg = pfit[4]

    _, y_complex = simulations_ptr(frequency_vector, k2, alfa2, r32, k3, **phys_params)
    y_model_norm = (y_complex / y_complex[0]) * np.exp(1j * np.deg2rad(phi0_deg))
    model_phase_deg = np.unwrap(np.angle(y_model_norm)) * 180 / np.pi

    gain = np.mean(exp_amp[:5] / np.abs(y_model_norm[:5]))
    model_amp_scaled = np.abs(y_model_norm) * gain

    exp_complex = exp_amp * np.exp(1j * np.deg2rad(exp_phase))
    exp_norm = (exp_complex / exp_complex[0]) * np.exp(1j * np.deg2rad(phi0_deg))
    exp_phase_deg_plot = np.unwrap(np.angle(exp_norm)) * 180 / np.pi

    result = PTRFitResult(
        k2=k2,
        alfa2=alfa2,
        r32=r32,
        k3=k3,
        phi0_deg=phi0_deg,
        res_norm=best_resnorm,
        model_amp=model_amp_scaled,
        exp_amp=exp_amp,
        model_phase_deg=model_phase_deg,
        exp_phase_deg=exp_phase_deg_plot,
        pfit=pfit,
        exit_flag=int(best_result.status),
        frequency_vector=frequency_vector,
        l2=phys_params.get('l2', 469e-9),

        n_starts=n_starts,
        best_resnorm=best_resnorm,
        all_results=all_results,
    )

    return result