import dataclasses
import numpy as np
import multiprocessing as mp
from functools import lru_cache

from joblib import Parallel, delayed
from scipy.optimize import least_squares

from app.methods.mulitilayer.simulations import simulate_single_frequency
from app.models.ptr_config import PTRConfig
from app.models.ptr_fit_result import PTRFitResult


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



@lru_cache(maxsize=1024)
def simulate_single_frequency_cached(omega: float, k2: float, alfa2: float,
                                     r32: float, k3: float,
                                     config_tuple: tuple, anisotropy: float) -> complex:
    """Wersja cache'owana - config zamieniany na tuple"""
    config = PTRConfig(**dict(config_tuple))  # odtwarzamy obiekt
    return simulate_single_frequency(omega, k2, alfa2, r32, k3, config, anisotropy)


def worker(task):
    """Worker"""
    i, p0, lb, ub, freq_hz, exp_amp, exp_phase_rad, config_dict, fixed_aniso = task
    config = PTRConfig(**config_dict)

    res = least_squares(
        ptr_residual,
        p0,
        bounds=(lb, ub),
        args=(freq_hz, exp_amp, exp_phase_rad, config, fixed_aniso),
        max_nfev=350,
        ftol=1e-9,
        xtol=1e-9,
        gtol=1e-9,
        verbose=0
    )
    print(f"   ✓ Worker {i + 1:2d} | koszt = {res.cost:.6e}", flush=True)
    return i, res


def ptr_residual(p, freq_hz, exp_amp, exp_phase_rad, config, fixed_aniso):
    logk2, logalfa2, logr32, logA, phi = p
    k2 = 10 ** logk2
    alfa2 = 10 ** logalfa2
    r32 = 10 ** logr32
    A = 10 ** logA

    omega = 2 * np.pi * freq_hz

    # Przygotowanie tupla z config (hashowalne)
    config_tuple = tuple(sorted(dataclasses.asdict(config).items()))

    T_surf = np.array([
        simulate_single_frequency_cached(w, k2, alfa2, r32, config.k3, config_tuple, fixed_aniso)
        for w in omega
    ])

    norm_factor = A * np.exp(-1j * phi) * np.sqrt(freq_hz)
    y_norm = T_surf * norm_factor

    amp_res = (np.abs(y_norm) - exp_amp) / (exp_amp + 1e-12)

    phase_diff = np.angle(y_norm) - exp_phase_rad
    phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))

    return np.concatenate([amp_res, phase_diff * config.phase_weight])


def fit_ptr_3d(freq_hz, exp_amp, exp_phase_deg, config: PTRConfig,
               n_starts=25, max_workers=None):
    exp_phase_rad = np.deg2rad(exp_phase_deg)
    fixed_aniso = config.anisotropy

    lb = np.array([np.log10(0.02), np.log10(1e-9), np.log10(1e-9), -10.0, -2 * np.pi])
    ub = np.array([np.log10(1.0), np.log10(1e-4), np.log10(1e-6), 10.0, 2 * np.pi])

    p0_expected = np.array([
        np.log10(0.18), np.log10(1.3e-7), np.log10(1e-8),
        np.log10(1e-5), np.deg2rad(-320)
    ])

    config_dict = dataclasses.asdict(config)

    tasks = []
    for i in range(n_starts):
        if i == 0:
            p0 = p0_expected.copy()
        else:
            p0 = p0_expected + np.array([
                np.random.uniform(-0.2, 0.2),
                np.random.uniform(-0.4, 0.4),
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-0.5, 0.5)
            ])
            p0 = np.clip(p0, lb, ub)

        tasks.append((
            i, p0, lb, ub,
            freq_hz.copy(), exp_amp.copy(), exp_phase_rad.copy(),
            config_dict, fixed_aniso
        ))

    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 2)

    print(f"Uruchamiam {n_starts} startów na {max_workers} rdzeniach...\n")

    parallel_results = Parallel(
        n_jobs=max_workers,
        backend='multiprocessing',
        verbose=0
    )(
        delayed(worker)(task) for task in tasks
    )

    # Najlepszy wynik
    best_cost = np.inf
    best_res = None
    for _, res in parallel_results:
        if res.cost < best_cost:
            best_cost = res.cost
            best_res = res

    print(f"Najlepszy koszt: {best_cost:.6e}\n")

    # Końcowy model
    k2 = 10 ** best_res.x[0]
    alfa2 = 10 ** best_res.x[1]
    r32 = 10 ** best_res.x[2]
    A = 10 ** best_res.x[3]
    phi = best_res.x[4]

    omega = 2 * np.pi * freq_hz
    config_tuple = tuple(sorted(dataclasses.asdict(config).items()))

    T_surf = np.array([
        simulate_single_frequency_cached(w, k2, alfa2, r32, config.k3, config_tuple, fixed_aniso)
        for w in omega
    ])

    y_final = A * np.exp(-1j * phi) * np.sqrt(freq_hz) * T_surf
    model_amp = np.abs(y_final)
    model_phase_deg = np.rad2deg(np.angle(y_final))

    return PTRFitResult(
        k2=k2, alfa2=alfa2, r32=r32, k3=config.k3,
        anisotropy=fixed_aniso, k_parallel=k2 * fixed_aniso,
        res_norm=best_cost,
        r2_amp=calculate_r2_amp(exp_amp, model_amp),
        r2_phase=calculate_r2(exp_phase_deg, model_phase_deg),
        model_amp=model_amp, model_phase_deg=model_phase_deg,
        exp_amp=exp_amp, exp_phase_deg=exp_phase_deg,
        frequency_hz=freq_hz,
    )