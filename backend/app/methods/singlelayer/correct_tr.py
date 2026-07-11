import numpy as np


def correct_tr_data(time_ns, signal):
    """
    Przyjmuje czas w nanosekundach i surowy sygnał fotodetiody.
    Zwraca czas w sekundach i oczyszczony sygnał.
    """
    time_s = time_ns * 1e-9

    # Baseline correction: odejmij średnią z pierwszych 5% punktów (przed strzałem)
    baseline_pts = max(5, len(signal) // 20)
    baseline = np.mean(signal[:baseline_pts])
    corrected_signal = signal - baseline

    # Filtracja medianowa dla szumów szpilkowych
    if len(corrected_signal) > 15:
        from scipy.signal import medfilt
        corrected_signal = medfilt(corrected_signal, kernel_size=5)

    return time_s, corrected_signal