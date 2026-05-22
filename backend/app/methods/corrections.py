import numpy as np
from scipy.signal import medfilt

def correct_ptr_data(freq_khz: np.ndarray, amp: np.ndarray, phase_deg: np.ndarray, max_freq: float = 2000):
    """
    Przygotowanie danych eksperymentalnych.

    KROK 1 (wg promotora): Faza ze stopni przeliczona na radiany (wewnątrz funkcji)
    - tutaj również wykonujemy unwrap i usuwanie outlierów, ale **kluczowa konwersja
      deg -> rad zostanie wykorzystana później przy tworzeniu liczby zespolonej**.
    """
    freq_hz = freq_khz * 1000.0

    # Konwersja na radiany do poprawnego unwrapowania
    phase_rad = np.deg2rad(phase_deg)
    phase_unwrapped = np.unwrap(phase_rad)
    phase_corr_deg = np.rad2deg(phase_unwrapped)   # faza ciągła w stopniach

    # Usuń ewidentne outliery w amplitudzie
    if len(amp) > 10:
        window = min(7, len(amp) // 2 * 2 + 1)
        amp_smooth = medfilt(amp, kernel_size=window)
        rel_diff = np.abs(amp - amp_smooth) / np.maximum(amp_smooth, 1e-12)
        mask = rel_diff < 0.08
        freq_hz = freq_hz[mask]
        amp = amp[mask]
        phase_corr_deg = phase_corr_deg[mask]

    # Ogranicz maksymalną częstotliwość
    mask = freq_hz <= max_freq * 1000
    freq_hz = freq_hz[mask]
    amp = amp[mask]
    phase_corr_deg = phase_corr_deg[mask]

    return freq_hz, amp, phase_corr_deg