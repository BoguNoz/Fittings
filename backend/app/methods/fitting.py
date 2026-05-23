import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc
from tqdm import tqdm

from .simulations import simulate_single_frequency
from ..models.ptr_fit_result import PTRFitResult


def ptr_residual(p, freq_hz, exp_amp, exp_phase_deg, config):
    """
    KROK 2: Obliczenie części rzeczywistej i urojonej EKSPERYMENTU.
    KROK 3: Obliczenie części rzeczywistej i urojonej MODELU.
    KROK 4: Normalizacja AMPLITUDY modelu: ampl /= sqrt(freq).
    KROK 5: Normalizacja FAZY modelu: faza -= 45°.
    """
    logk2, log_aniso, logr32, logk3 = p
    k2 = 10**logk2
    anisotropy = 10**log_aniso
    r32 = 10**logr32
    k3_fit = 10**logk3
    alfa2 = k2 / config.rhoc2

    # --- KROK 3: Surowy zespolony sygnał modelu (brak normalizacji) ---
    y_model = np.array([
        simulate_single_frequency(2*np.pi*f, k2, alfa2, r32, k3_fit, config, anisotropy)
        for f in freq_hz
    ])

    # --- KROK 4 i 5: Normalizacja modelu (tylko model!) ---
    # amplituda /= sqrt(freq), faza -= 45°
    norm_factor = np.exp(-1j * np.deg2rad(45.0)) * np.sqrt(freq_hz)
    y_norm = y_model * norm_factor

    # --- KROK 2: Zespolony sygnał eksperymentalny ---
    exp_complex = exp_amp * np.exp(1j * np.deg2rad(exp_phase_deg))

    # Ważenie (opcjonalne)
    w = (freq_hz / freq_hz.max()) ** config.weight_exponent
    diff = (y_norm - exp_complex) / np.maximum(np.abs(exp_complex), 1e-12)
    diff = diff * w

    residuals = np.concatenate([diff.real, config.phase_weight * diff.imag])
    return residuals #


def fit_ptr_3d(freq_hz, exp_amp, exp_phase_deg, config, n_starts=25):
    """
    KROK 6: Proces fitowania (wielostartowy).
    KROK 7: Po zakończeniu fitowania wykorzystujemy TEN SAM model
            (drugi model) z optymalnymi parametrami do wygenerowania
            końcowych krzywych amplitudy i fazy.
    """
    lb = np.array([np.log10(0.05), np.log10(1.0), np.log10(1e-9), np.log10(0.3)])
    ub = np.array([np.log10(1.5),  np.log10(10.0), np.log10(1e-5), np.log10(3.0)])

    best_res = None
    best_cost = np.inf

    print(f"Starting multi-start optimization with {n_starts} attempts...")
    for i in tqdm(range(n_starts), desc="Multi-start optimization"):
        if i == 0:
            p0 = np.array([np.log10(0.25), np.log10(2.5), np.log10(2e-7), np.log10(1.0)])
        else:
            sampler = qmc.LatinHypercube(d=4, seed=42 + i)
            sample = sampler.random(1)[0]
            p0 = lb + sample * (ub - lb)

        try:
            res = least_squares(ptr_residual, p0, bounds=(lb, ub),
                                args=(freq_hz, exp_amp, exp_phase_deg, config),
                                max_nfev=1500, ftol=1e-9, xtol=1e-9, verbose=0)
            if res.cost < best_cost and np.isfinite(res.cost):
                best_cost = res.cost
                best_res = res
        except Exception:
            continue

    if best_res is None:
        raise RuntimeError("All optimization attempts failed!")

    # Rekonstrukcja parametrów
    k2 = 10 ** best_res.x[0]
    anisotropy = 10 ** best_res.x[1]
    r32 = 10 ** best_res.x[2]
    k3_fit = 10 ** best_res.x[3]
    alfa2 = k2 / config.rhoc2
    k_parallel = k2 * anisotropy

    # --- KROK 7: Drugi model – generowanie końcowych krzywych ---
    y_final = np.array([
        simulate_single_frequency(2*np.pi*f, k2, alfa2, r32, k3_fit, config, anisotropy)
        for f in freq_hz
    ])
    # Stosujemy tę samą normalizację co przy fitowaniu
    norm_factor = np.exp(-1j * np.deg2rad(45.0)) * np.sqrt(freq_hz)
    y_final = y_final * norm_factor

    # # --- KROK 7: Drugi model – generowanie końcowych krzywych ---
    # y_final = np.array([
    #     simulate_ptr_single_frequency(2 * np.pi * f, k2, alfa2, r32, k3_fit, config)
    #     for f in freq_hz
    # ])
    #
    # # Stosujemy tę samą normalizację co przy fitowaniu
    # norm_factor = np.exp(-1j * np.deg2rad(45.0)) / np.sqrt(freq_hz)
    # y_final = y_final * norm_factor

    print(f"\nBest fit achieved with cost = {best_cost:.2e}")

    return PTRFitResult(
        k2=k2,
        alfa2=alfa2,
        r32=r32,
        k3=k3_fit,
        anisotropy=anisotropy,
        k_parallel=k_parallel,
        res_norm=best_cost,
        model_amp=np.abs(y_final),
        model_phase_deg=np.angle(y_final, deg=True),
        exp_amp=exp_amp,
        exp_phase_deg=exp_phase_deg,
        frequency_hz=freq_hz
    )