import numpy as np
from scipy.signal import medfilt

def correct_ptr_data(
    freq: np.ndarray,
    amp: np.ndarray,
    phase_deg: np.ndarray,
    remove_amp_outliers: bool = True,
    outlier_threshold: float = 0.05
):
    """
    Correct raw PTR data: unwrap phase and optionally remove amplitude outliers.

    Phase unwrapping is performed to convert wrapped (e.g., -180°..+180°) data
    into a continuous monotonic function. Amplitude outliers are detected by
    comparing each point with a median-filtered (smoothed) version of the amplitude.

    Parameters
    ----------
    freq : np.ndarray
        Frequency vector [Hz].
    amp : np.ndarray
        Raw amplitude data.
    phase_deg : np.ndarray
        Raw phase in degrees (may be wrapped).
    remove_amp_outliers : bool, optional
        If True, points with amplitude deviating more than `outlier_threshold`
        from the local median trend are removed.
    outlier_threshold : float, optional
        Relative deviation threshold for outlier removal (default 5 %).

    Returns
    -------
    freq_corr : np.ndarray
        Corrected frequency vector.
    amp_corr : np.ndarray
        Corrected amplitude vector.
    phase_corr_deg : np.ndarray
        Unwrapped phase in degrees (continuous).
    """
    # Phase unwrapping
    phase_rad = np.deg2rad(phase_deg)
    phase_unwrapped_rad = np.unwrap(phase_rad)
    phase_corr_deg = np.rad2deg(phase_unwrapped_rad)

    # Amplitude outlier removal
    if remove_amp_outliers and len(amp) > 5:
        window = min(5, len(amp) // 2 * 2 + 1)  # odd, max 5
        if window >= 3:
            amp_smooth = medfilt(amp, kernel_size=window)
        else:
            amp_smooth = amp

        relative_diff = np.abs(amp - amp_smooth) / np.maximum(amp_smooth, 1e-12)
        mask = relative_diff < outlier_threshold

        if np.sum(mask) >= 5:   # keep at least 5 points
            freq_corr = freq[mask]
            amp_corr = amp[mask]
            phase_corr_deg = phase_corr_deg[mask]
        else:
            freq_corr = freq
            amp_corr = amp
            # phase_corr_deg already unwrapped
    else:
        freq_corr = freq
        amp_corr = amp

    return freq_corr, amp_corr, phase_corr_deg