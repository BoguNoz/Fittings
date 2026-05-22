import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc
from tqdm import tqdm
from dataclasses import dataclass

from .simulations import simulate_single_frequency
from ..models.ptr_fit_result import PTRFitResult

def ptr_residual(p, freq_hz, exp_amp, exp_phase_deg, config):
    """
    Oblicza residua dla jednego zestawu parametrów.

    Zawiera **kroki 2, 3 i 4** instrukcji promotora:
    KROK 2: Eksperyment -> część rzeczywista i urojona (exp_complex)
    KROK 3: Model -> część rzeczywista i urojona (y_model)
    KROK 4: Normalizacja modelu: amplituda = amplituda / sqrt(f)  oraz  faza = faza - 45°
    """
    # Rozpakowanie parametrów (w skali log10, bez phi0)
    logk2, log_aniso, logr32, logk3 = p

    k2 = 10**logk2
    anisotropy = 10**log_aniso
    r32 = 10**logr32
    k3_fit = 10**logk3
    alfa2 = k2 / config.rhoc2

    # ---------- KROK 3: część rzeczywista i urojona modelu ----------
    y_model = np.array([
        simulate_single_frequency(2 * np.pi * f, k2, alfa2, r32, k3_fit, config, anisotropy)
        for f in freq_hz
    ])   # sygnał zespolony z modelu Henkela

    # ---------- KROK 4: normalizacja modelu wg wzoru promotora ----------
    # amplituda / sqrt(freq)  oraz  faza - 45°
    norm_factor = np.exp(-1j * np.deg2rad(45.0)) / np.sqrt(freq_hz)
    y_norm = y_model * norm_factor

    # ---------- KROK 2: część rzeczywista i urojona eksperymentu ----------
    exp_complex = exp_amp * np.exp(1j * np.deg2rad(exp_phase_deg))

    # Ważenie zależne od częstotliwości (opcjonalne, zachowane z oryginału)
    w = (freq_hz / freq_hz.max()) ** config.weight_exponent
    diff = (y_norm - exp_complex) / np.maximum(np.abs(exp_complex), 1e-12)
    diff = diff * w

    # Residua: część rzeczywista + ważona część urojona
    residuals = np.concatenate([diff.real, config.phase_weight * diff.imag])
    return residuals

def fit_ptr_3d(
        freq_hz: np.ndarray,
        exp_amp: np.ndarray,
        exp_phase_deg: np.ndarray,
        config,
        n_starts: int = 25
):
    """
    KROK 5: Proces fitowania (wielostartowa optymalizacja least_squares).
    Parametry: logk2, log_aniso, logr32, logk3  (bez dodatkowego przesunięcia fazy).
    """
    # Granice w log10
    lb = np.array([np.log10(0.05), np.log10(1.0), np.log10(1e-9), np.log10(0.3)])
    ub = np.array([np.log10(1.5),  np.log10(10.0), np.log10(1e-5), np.log10(3.0)])

    best_res = None
    best_cost = np.inf

    print(f"Starting multi-start optimization with {n_starts} attempts...")

    for i in tqdm(range(n_starts), desc="Multi-start optimization"):
        if i == 0:
            # start z ręcznie podanych wartości
            p0 = np.array([np.log10(0.25), np.log10(2.5), np.log10(2e-7), np.log10(1.0)])
        else:
            sampler = qmc.LatinHypercube(d=4, seed=42 + i)
            sample = sampler.random(1)[0]
            p0 = lb + sample * (ub - lb)

        try:
            res = least_squares(
                ptr_residual,
                p0,
                bounds=(lb, ub),
                args=(freq_hz, exp_amp, exp_phase_deg, config),
                max_nfev=1500,
                ftol=1e-9,
                xtol=1e-9,
                verbose=0
            )

            if res.cost < best_cost and np.isfinite(res.cost):
                best_cost = res.cost
                best_res = res

        except Exception as e:
            print(f"  Attempt {i} failed: {type(e).__name__}")
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

    # ---------- KROK 6: Drugi model – generowanie amplitudy i fazy po fitowaniu ----------
    # Używamy tego samego modelu Henkela z dopasowanymi parametrami
    y_model_final = np.array([
        simulate_single_frequency(2 * np.pi * f, k2, alfa2, r32, k3_fit, config, anisotropy)
        for f in freq_hz
    ])

    # Stosujemy taką samą normalizację (amplituda/sqrt(f), faza-45°) aby otrzymać końcowe krzywe
    norm_factor = np.exp(-1j * np.deg2rad(45.0)) / np.sqrt(freq_hz)
    y_final_norm = y_model_final * norm_factor

    print(f"\nBest fit achieved with cost = {best_cost:.2e}")

    return PTRFitResult(
        k2=k2,
        alfa2=alfa2,
        r32=r32,
        k3=k3_fit,
        anisotropy=anisotropy,
        k_parallel=k_parallel,
        res_norm=best_cost,
        model_amp=np.abs(y_final_norm),               # amplituda po normalizacji
        model_phase_deg=np.angle(y_final_norm, deg=True),  # faza po normalizacji
        exp_amp=exp_amp,
        exp_phase_deg=exp_phase_deg,
        frequency_hz=freq_hz
    )