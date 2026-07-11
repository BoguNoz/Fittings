import numpy as np
from scipy.signal import medfilt

def correct_ptr_data(freq_khz, amp, phase_deg, max_freq=2000):
    """
    KROK 1: Faza ze stopni na radiany (wewnętrznie do unwrapowania).
    Zwracamy fazę w stopniach (już po rozwinięciu), bo później i tak
    przy tworzeniu liczby zespolonej użyjemy np.deg2rad().
    """
    freq_hz = freq_khz * 1000.0

    # unwrap po konwersji na radiany
    phase_rad = np.deg2rad(phase_deg)
    phase_unwrapped = np.unwrap(phase_rad)
    phase_deg_corr = np.rad2deg(phase_unwrapped)

    # filtracja odstających punktów
    if len(amp) > 10:
        window = min(7, len(amp) // 2 * 2 + 1)
        amp_smooth = medfilt(amp, kernel_size=window)
        rel_diff = np.abs(amp - amp_smooth) / np.maximum(amp_smooth, 1e-12)
        mask = rel_diff < 0.08
        freq_hz = freq_hz[mask]
        amp = amp[mask]
        phase_deg_corr = phase_deg_corr[mask]

    mask = freq_hz <= max_freq * 1000
    freq_hz = freq_hz[mask]
    amp = amp[mask]
    phase_deg_corr = phase_deg_corr[mask]

    return freq_hz, amp, phase_deg_corr