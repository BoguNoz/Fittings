import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm
from .simulations import simulate_single_frequency
from ..models.ptr_fit_result import PTRFitResult


def calculate_r2(exp, model):
    """Klasyczne R^2 (dobre dla fazy)"""
    ss_res = np.sum((exp - model) ** 2)
    ss_tot = np.sum((exp - np.mean(exp)) ** 2)
    return 1 - (ss_res / ss_tot)


def calculate_r2_amp(exp, model):
    """Poprawione R^2 w skali logarytmicznej dla amplitudy (zapobiega dominacji niskich częstotliwości)"""
    exp_log = np.log10(exp + 1e-12)
    model_log = np.log10(model + 1e-12)
    ss_res = np.sum((exp_log - model_log) ** 2)
    ss_tot = np.sum((exp_log - np.mean(exp_log)) ** 2)
    return 1 - (ss_res / ss_tot)


def ptr_residual(p, freq_hz, exp_amp, exp_phase_rad, config, fixed_aniso):
    """
    p = [log10(k2), log10(alfa2), log10(r32), log10(A), phi]

    alfa2 jest teraz niezależną zmienną decyzyjną (nie zależy od k2/rhoc2).
    """
    logk2, logalfa2, logr32, logA, phi = p
    k2 = 10 ** logk2
    alfa2 = 10 ** logalfa2  # Wyciągamy niezależną dyfuzyjność
    r32 = 10 ** logr32
    A = 10 ** logA

    omega = 2 * np.pi * freq_hz
    # Przekazujemy niezależne alfa2 bezpośrednio do symulacji
    T_surf = np.array([
        simulate_single_frequency(w, k2, alfa2, r32, config.k3, config, fixed_aniso)
        for w in omega
    ])

    # Normalizacja aparaturowa: A * exp(-i*phi) * sqrt(f)
    norm_factor = A * np.exp(-1j * phi) * np.sqrt(freq_hz)
    y_norm = T_surf * norm_factor

    # Residua amplitudy – względne
    amp_res = (np.abs(y_norm) - exp_amp) / (exp_amp + 1e-12)

    # Residua fazy – zawinięte do dziedziny [-π, π]
    phase_diff = np.angle(y_norm) - exp_phase_rad
    phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
    phase_res = phase_diff

    return np.concatenate([amp_res, phase_res * config.phase_weight])


def fit_ptr_3d(freq_hz, exp_amp, exp_phase_deg, config, n_starts=25):
    exp_phase_rad = np.deg2rad(exp_phase_deg)
    fixed_aniso = config.anisotropy

    # Nowe granice fizyczne (5 parametrów): [log10(k2), log10(alfa2), log10(r32), log10(A), phi]
    # Dla alfa2 dopuszczamy zakres od 1e-9 do 1e-4 m²/s (typowy szeroki zakres dla ciał stałych i cienkich warstw)
    lb = np.array([np.log10(0.02), np.log10(1e-9), np.log10(1e-9), -10.0, -2 * np.pi])
    ub = np.array([np.log10(1.0), np.log10(1e-4), np.log10(1e-6), 10.0, 2 * np.pi])

    # Oczekiwane wartości początkowe (punkt startowy nr 1)
    k2_exp = 0.18
    alfa2_exp = 1.3e-7  # Szacowana wartość oczekiwana dla warstwy (np. k2_exp / rhoc2)
    r32_exp = 1e-8
    A_exp = 1e-5
    phi_exp = np.deg2rad(-320)

    p0_expected = np.array([np.log10(k2_exp), np.log10(alfa2_exp), np.log10(r32_exp), np.log10(A_exp), phi_exp])

    best_res = None
    best_cost = np.inf

    for i in tqdm(range(n_starts), desc="Optymalizacja 5D (Multi-start)"):
        if i == 0:
            p0 = p0_expected
        else:
            # Losowanie wokół wartości oczekiwanych z odpowiednim rozrzutem w skali log
            p0 = p0_expected + np.array([
                np.random.uniform(-0.2, 0.2),  # logk2
                np.random.uniform(-0.4, 0.4),  # logalfa2 (nowe!)
                np.random.uniform(-0.5, 0.5),  # logr32
                np.random.uniform(-1.0, 1.0),  # logA
                np.random.uniform(-0.5, 0.5)  # phi (rad)
            ])
            # Przycięcie do twardych granic lb i ub
            p0 = np.clip(p0, lb, ub)

        res = least_squares(
            ptr_residual, p0, bounds=(lb, ub),
            args=(freq_hz, exp_amp, exp_phase_rad, config, fixed_aniso),
            max_nfev=400,  # Zwiększone do 400 z uwagi na wyższy wymiar przestrzeni (5D)
        )
        if res.cost < best_cost:
            best_cost, best_res = res.cost, res

    # Odtworzenie ostatecznych parametrów z najlepszego dopasowania
    k2 = 10 ** best_res.x[0]
    alfa2 = 10 ** best_res.x[1]
    r32 = 10 ** best_res.x[2]
    A = 10 ** best_res.x[3]
    phi = best_res.x[4]

    # Generowanie końcowego modelu symulacyjnego dla wyznaczonych parametrów
    omega = 2 * np.pi * freq_hz
    T_surf = np.array([
        simulate_single_frequency(w, k2, alfa2, r32, config.k3, config, fixed_aniso)
        for w in omega
    ])
    y_final = A * np.exp(-1j * phi) * np.sqrt(freq_hz) * T_surf
    model_amp = np.abs(y_final)
    model_phase_deg = np.rad2deg(np.angle(y_final))

    return PTRFitResult(
        k2=k2, alfa2=alfa2, r32=r32, k3=config.k3,
        anisotropy=fixed_aniso, k_parallel=k2 * fixed_aniso,
        res_norm=best_cost,
        r2_amp=calculate_r2_amp(exp_amp, model_amp),  # Zastosowanie log-R^2
        r2_phase=calculate_r2(exp_phase_deg, model_phase_deg),
        model_amp=model_amp, model_phase_deg=model_phase_deg,
        exp_amp=exp_amp, exp_phase_deg=exp_phase_deg, frequency_hz=freq_hz,
    )