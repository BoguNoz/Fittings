import numpy as np
from scipy.signal import medfilt

def correct_ptr_data(
    freq: np.ndarray,
    amp: np.ndarray,
    phase_deg: np.ndarray,
    remove_amp_outliers: bool = True
):
    """Correct phase (unwrap) and optionally remove amplitude outliers."""
    # Unwrap phase
    phase_rad = np.deg2rad(phase_deg)
    phase_unwrapped = np.unwrap(phase_rad)
    phase_corr_deg = np.rad2deg(phase_unwrapped)

    if remove_amp_outliers and len(amp) > 10:
        window = min(7, len(amp) // 2 * 2 + 1)
        amp_smooth = medfilt(amp, kernel_size=window)
        rel_diff = np.abs(amp - amp_smooth) / np.maximum(amp_smooth, 1e-12)
        mask = rel_diff < 0.08  # 8% threshold
        if np.sum(mask) >= 8:
            freq = freq[mask]
            amp = amp[mask]
            phase_corr_deg = phase_corr_deg[mask]

    return freq, amp, phase_corr_deg