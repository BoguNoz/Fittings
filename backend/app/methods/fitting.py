import numpy as np
from scipy.optimize import least_squares
from scipy.stats import qmc
from tqdm import tqdm
from .simulations import simulate_single_frequency
from ..models.ptr_fit_result import PTRFitResult


def calculate_r2(exp, model):
    """Pomocnicza funkcja do obliczenia R^2."""
    ss_res = np.sum((exp - model) ** 2)
    ss_tot = np.sum((exp - np.mean(exp)) ** 2)
    return 1 - (ss_res / ss_tot)


def ptr_residual(p, freq_hz, exp_amp, exp_phase_deg, config, fixed_aniso):
    # Logarytmiczne skalowanie parametrów dla stabilności optymalizacji
    logk2, logr32, scale = p
    k2 = 10**logk2
    r32 = 10**logr32
    alfa2 = k2 / config.rhoc2

    # Obliczenie modelu bez paska postępu
    y_model = np.array([
        simulate_single_frequency(2 * np.pi * f, k2, alfa2, r32, config.k3, config, fixed_aniso)
        for f in freq_hz
    ])

    # Normalizacja modelu
    norm_factor = np.exp(-1j * np.deg2rad(45.0)) * np.sqrt(freq_hz)

    scale = 10 ** scale
    y_norm = y_model * norm_factor * scale

    # Konwersja eksperymentu na postać zespoloną
    exp_complex = exp_amp * np.exp(1j * np.deg2rad(exp_phase_deg))

    # Obliczenie residuów (błąd względny dla stabilności)
    diff = (y_norm - exp_complex) / np.maximum(np.abs(exp_complex), 1e-12)
    residuals = np.concatenate([diff.real, config.phase_weight * diff.imag])

    # print(f"Model max: {np.max(np.abs(y_norm))}, Exp max: {np.max(np.abs(exp_complex))}")
    # # Tuż przed return residuals
    # print(f"DEBUG: p={p}, logk2={p[0]}, logr32={p[1]}")
    # Sprawdź jeden punkt:
    test_val = simulate_single_frequency(2 * np.pi * freq_hz[0], 10 ** p[0], (10 ** p[0]) / config.rhoc2, 10 ** p[1],
                                         config.k3, config, fixed_aniso)
    #print(f"DEBUG: Pojedyncza symulacja zwraca: {test_val}")
    return residuals


def fit_ptr_3d(freq_hz, exp_amp, exp_phase_deg, config, n_starts=25):
    """
    KROK 6: Wielostartowa optymalizacja (Globalna).
    KROK 7: Rekonstrukcja końcowych parametrów i statystyk (R^2).
    """
    # Stała anizotropia (np. 3.67 dla PEDOT:PSS z literatury)
    fixed_aniso = config.anisotropy

    # Granice dla k2 i r32
    lb = np.array([np.log10(0.05), np.log10(1e-9), -10])
    ub = np.array([np.log10(1.5), np.log10(1e-5), -6])

    best_res = None
    best_cost = np.inf

    for i in tqdm(range(n_starts), desc="Optymalizacja (Multi-start)"):
        p0 = lb + (ub - lb) * qmc.LatinHypercube(d=3, seed=42 + i).random(1)[0]

        res = least_squares(ptr_residual, p0, bounds=(lb, ub),
                            args=(freq_hz, exp_amp, exp_phase_deg, config, fixed_aniso),
                            max_nfev=100, ) # Ogranicz liczbę kroków
                            # verbose=2)  # Wypisze na ekranie, jak zmienia się koszt

        if res.cost < best_cost:
            best_cost, best_res = res.cost, res

    # --- KROK 7: Rekonstrukcja końcowa ---
    k2 = 10 ** best_res.x[0]
    r32 = 10 ** best_res.x[1]
    y_final = np.array(
        [simulate_single_frequency(2 * np.pi * f, k2, k2 / config.rhoc2, r32, config.k3, config, fixed_aniso) for f in
         freq_hz])
    y_final *= (np.exp(-1j * np.deg2rad(45.0)) * np.sqrt(freq_hz))

    return PTRFitResult(
        k2=k2, alfa2=k2 / config.rhoc2, r32=r32, k3=config.k3,
        anisotropy=fixed_aniso, k_parallel=k2 * fixed_aniso,
        res_norm=best_cost,
        r2_amp=calculate_r2(exp_amp, np.abs(y_final)),
        r2_phase=calculate_r2(exp_phase_deg, np.angle(y_final, deg=True)),
        model_amp=np.abs(y_final), model_phase_deg=np.angle(y_final, deg=True),
        exp_amp=exp_amp, exp_phase_deg=exp_phase_deg, frequency_hz=freq_hz
    )