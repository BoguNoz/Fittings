import numpy as np
from scipy.optimize import least_squares
from joblib import Parallel, delayed
import multiprocessing as mp

from app.methods.singlelayer.simulate_tr import simulate_tr_single_layer


def tr_residual(p, time_s, exp_signal, config):
    """
    p = [log10(k), log10(alfa), log10(A), t0]
    """
    logk, logalfa, logA, t0 = p
    k = 10 ** logk
    alfa = 10 ** logalfa
    A = 10 ** logA

    # Przesunięcie czasu (time zero correction)
    shifted_time = time_s - t0

    # Sygnał istnieje tylko dla t > 0
    mask = shifted_time > 0
    model_signal = np.zeros_like(time_s)

    if np.any(mask):
        model_signal[mask] = A * simulate_tr_single_layer(shifted_time[mask], k, alfa, config)

    # Klasyczny rezyduum (błąd bezwzględny lub względny)
    return model_signal - exp_signal


def fit_tr_1d(time_s, exp_signal, config, n_starts=15, max_workers=None):
    """Multi-start dla dopasowania modelu TR jednej warstwy"""

    # Granice dla: log10(k), log10(alfa), log10(A), t0
    lb = np.array([np.log10(0.1), np.log10(1e-7), np.log10(1e-8), -1e-9])
    ub = np.array([np.log10(500.0), np.log10(2e-4), np.log10(1e2), 1e-9])

    # Punkt startowy (oczekiwany np. dla krzemu/metalu)
    p0_expected = np.array([np.log10(10.0), np.log10(1e-5), np.log10(1e-3), 0.0])

    tasks = []
    for i in range(n_starts):
        if i == 0:
            p0 = p0_expected.copy()
        else:
            p0 = p0_expected + np.array([
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(-1.0, 1.0),
                np.random.uniform(-2e-10, 2e-10)
            ])
            p0 = np.clip(p0, lb, ub)

        tasks.append((i, p0, lb, ub, time_s.copy(), exp_signal.copy(), config))

    if max_workers is None:
        max_workers = max(1, mp.cpu_count() - 2)

    def worker_tr(task):
        idx, p0_w, low, upp, t_w, exp_w, conf = task
        res = least_squares(
            tr_residual, p0_w, bounds=(low, upp),
            args=(t_w, exp_w, conf),
            max_nfev=400, ftol=1e-9, xtol=1e-9
        )
        return idx, res

    parallel_results = Parallel(n_jobs=max_workers)(
        delayed(worker_tr)(t) for t in tasks
    )

    # Wybór najlepszego dopasowania
    best_cost = np.inf
    best_res = None
    for _, res in parallel_results:
        if res.cost < best_cost:
            best_cost = res.cost
            best_res = res

    # Wyciągnięcie wyników końcowych
    final_k = 10 ** best_res.x[0]
    final_alfa = 10 ** best_res.x[1]
    final_A = 10 ** best_res.x[3]
    final_t0 = best_res.x[3]

    # Wygenerowanie finalnego modelu do wykresu
    shifted_time = time_s - final_t0
    mask = shifted_time > 0
    model_signal = np.zeros_like(time_s)
    if np.any(mask):
        model_signal[mask] = final_A * simulate_tr_single_layer(shifted_time[mask], final_k, final_alfa, config)

    # Obliczenie R^2
    ss_res = np.sum((exp_signal - model_signal) ** 2)
    ss_tot = np.sum((exp_signal - np.mean(exp_signal)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {
        "k": final_k,
        "alfa": final_alfa,
        "t0": final_t0,
        "r2": r2,
        "model_signal": model_signal,
        "cost": best_cost
    }