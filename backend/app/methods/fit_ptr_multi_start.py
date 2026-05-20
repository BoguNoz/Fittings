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
        exp_phase: np.ndarray,
        n_starts: int = 30,
        **phys_params
) -> PTRFitResult:
    best_result = None
    best_resnorm = np.inf
    all_results = []
    np.random.seed(42)

    low_freq_mode = frequency_vector.max() < 1e5
    if low_freq_mode:
        lower = [np.log10(0.05), np.log10(1.0),  np.log10(1e-10), np.log10(0.8)]
        upper = [np.log10(1.5),  np.log10(10.0), np.log10(1e-5),  np.log10(1.5)]
    else:
        lower = [np.log10(0.02), np.log10(1.0), np.log10(1e-10), np.log10(0.2)]
        upper = [np.log10(1.5),  np.log10(10.0), np.log10(1e-5),  np.log10(3.0)]

    rhoc = phys_params.get('rhoc', 2.0e6)

    for i in tqdm(range(n_starts), desc="Multi‑start fitting"):
        p0 = generate_initial_guess(i, n_starts, low_freq_mode)
        try:
            res = least_squares(
                ptr_residual,
                p0,
                bounds=(lower, upper),
                args=(frequency_vector, exp_amp, exp_phase),
                kwargs=phys_params,
                ftol=1e-8, xtol=1e-8, max_nfev=3000
            )
            cost = 2 * res.cost
            pfit = res.x
            k2 = 10 ** pfit[0]
            anisotropy = 10 ** pfit[1]
            r32 = 10 ** pfit[2]
            k3 = 10 ** pfit[3]
            alfa2 = k2 / rhoc
            k_parallel = k2 * anisotropy

            all_results.append({
                'pfit': pfit,
                'resnorm': cost,
                'k2': k2,
                'anisotropy': anisotropy,
                'r32': r32,
                'k3': k3,
                'alfa2': alfa2,
                'k_parallel': k_parallel,
                'status': res.status,
                'nfev': res.nfev
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
    alfa2 = k2 / rhoc
    k_parallel = k2 * anisotropy

    model_kwargs = {key: phys_params[key] for key in
                    ['k1', 'l1', 'l2', 'alfa1', 'alfa3', 'r21', 'd_pump', 'Q']
                    if key in phys_params}
    _, y_complex = simulations_ptr_hankel(
        frequency_vector, k2, alfa2, r32, k3,
        anisotropy=anisotropy, **model_kwargs
    )

    # ===== POPRAWKA: używamy tylko częstotliwości do 100 kHz =====
    exp_phase_rad = np.unwrap(np.deg2rad(exp_phase))  # jeszcze raz dla bezpieczeństwa
    exp_complex = exp_amp * np.exp(1j * exp_phase_rad)

    stable_mask = frequency_vector <= 100_000   # 100 kHz
    if np.any(stable_mask):
        G_opt = np.vdot(y_complex[stable_mask], exp_complex[stable_mask]) / \
                np.vdot(y_complex[stable_mask], y_complex[stable_mask])
    else:
        G_opt = np.vdot(y_complex, exp_complex) / np.vdot(y_complex, y_complex)

    model_scaled = G_opt * y_complex
    model_amp = np.abs(model_scaled)
    model_phase_deg = np.angle(model_scaled, deg=True)


    result = PTRFitResult(
        k2=k2,
        alfa2=alfa2,
        r32=r32,
        k3=k3,
        phi0_deg=np.angle(G_opt, deg=True), 
        anisotropy=anisotropy,
        k_parallel=k_parallel,
        res_norm=best_resnorm,
        model_amp=model_amp,
        exp_amp=exp_amp,
        model_phase_deg=model_phase_deg,
        exp_phase_deg=exp_phase,
        pfit=pfit,
        exit_flag=int(best_result.status),
        frequency_vector=frequency_vector,
        l2=phys_params.get('l2', 240e-9),
        n_starts=n_starts,
        best_resnorm=best_resnorm,
        all_results=all_results,
    )
    return result