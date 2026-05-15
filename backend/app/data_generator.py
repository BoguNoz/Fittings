import numpy as np
from app.methods.simulations_ptr import simulations_ptr

def generate_realistic_ptr_data(
    k2: float,
    alfa2: float,
    r32: float,
    k3: float,
    phi0_deg: float,
    freq_range: tuple = (1.0, 2000.0),
    n_points: int = 30,
    spacing: str = "log",
    *,
    # Noise and artefact parameters
    noise_amp_rel: float = 0.01,
    noise_phase_deg: float = 0.3,
    gain_drift_percent: float = 0.0,
    outlier_prob: float = 0.03,
    outlier_amp_factor: float = 0.85,
    wrap_phase: bool = True,
    # Fixed physical parameters
    k1: float = 200.0,
    l1: float = 50e-9,
    alfa1: float = 8.2e-5,
    r21: float = 1e-8,
    l2: float = 240e-9,
    alfa3: float = 0.5e-7,
    seed: int = None
):
    """
    Generate synthetic PTR data that mimics realistic experimental conditions.

    The function computes the exact PTR response for a 3‑layer system,
    then adds noise, optional systematic gain drift, phase wrapping,
    and isolated outliers to simulate real measurements.

    Parameters
    ----------
    k2, alfa2, r32, k3 : float
        True thermal parameters of layer 2 and 3 (and interface).
    phi0_deg : float
        True global phase offset in degrees.
    freq_range : tuple, optional
        (f_min, f_max) in Hz.
    n_points : int, optional
        Number of frequency points.
    spacing : str, optional
        'log' for logarithmic spacing, 'linear' for linear.
    noise_amp_rel : float, optional
        Relative 1‑σ noise of amplitude (normal distribution).
    noise_phase_deg : float, optional
        Phase noise standard deviation in degrees.
    gain_drift_percent : float, optional
        Total systematic amplitude drift (in %) from first to last point.
    outlier_prob : float, optional
        Probability of an isolated outlier at any point.
    outlier_amp_factor : float, optional
        Factor by which an outlier amplitude is multiplied (e.g. 0.85 = 15% drop).
    wrap_phase : bool, optional
        If True, the generated phase is wrapped into –180°..+180° as a lock‑in would.
    k1, l1, alfa1, r21, l2, alfa3 : float
        Fixed physical constants for the metal transducer, substrate, and interfaces.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    freq : np.ndarray
        Frequency vector [Hz].
    amp : np.ndarray
        Noisy, possibly drifted and perturbed amplitude.
    phase_deg : np.ndarray
        Noisy phase (possibly wrapped) in degrees.
    """
    if seed is not None:
        np.random.seed(seed)

    # Frequency grid
    if spacing == "log":
        freq = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_points)
    else:
        freq = np.linspace(freq_range[0], freq_range[1], n_points)

    # Exact model response
    _, y_complex = simulations_ptr(
        freq, k2, alfa2, r32, k3,
        k1=k1, l1=l1, alfa1=alfa1, alfa3=alfa3, r21=r21, l2=l2
    )
    # Normalisation and phase offset
    y_norm = (y_complex / y_complex[0]) * np.exp(1j * np.deg2rad(phi0_deg))
    amp_true = np.abs(y_norm)
    phase_true = np.unwrap(np.angle(y_norm)) * 180 / np.pi

    # Add random noise
    amp_noisy = amp_true * np.random.normal(1, noise_amp_rel, size=n_points)
    phase_noisy = phase_true + np.random.normal(0, noise_phase_deg, size=n_points)

    # Systematic gain drift (linear with frequency index)
    if gain_drift_percent != 0.0:
        drift = np.linspace(0, gain_drift_percent / 100.0, n_points)
        amp_noisy *= (1 + drift)

    # Phase wrapping (simulate lock‑in wrapping)
    if wrap_phase:
        phase_noisy = (phase_noisy + 180) % 360 - 180   # wrap to [-180, 180]

    # Isolated outliers (amplitude glitches)
    if outlier_prob > 0:
        outlier_mask = np.random.random(n_points) < outlier_prob
        amp_noisy[outlier_mask] *= outlier_amp_factor

    return freq, amp_noisy, phase_noisy


def save_ptr_data_to_file(
    filename: str,
    freq: np.ndarray,
    amp: np.ndarray,
    phase_deg: np.ndarray
):
    """
    Save PTR data to a text file with columns: freq, amplitude, phase.

    Parameters
    ----------
    filename : str
        Path to the output file.
    freq, amp, phase_deg : np.ndarray
        Data vectors.
    """
    with open(filename, 'w') as f:
        f.write('# freq_Hz   amplitude   phase_deg\n')
        for fr, am, ph in zip(freq, amp, phase_deg):
            f.write(f'{fr:.4f}    {am:.6f}    {ph:.6f}\n')


freq, amp, phase = generate_realistic_ptr_data(
    k2=0.22, alfa2=1.5e-7, r32=1.5e-7, k3=1.0, phi0_deg=5.0,
    n_points=35,
    noise_amp_rel=0.012,
    noise_phase_deg=0.4,
    gain_drift_percent=2.0,     # 2% amplitude drift across the band
    outlier_prob=0.05,
    wrap_phase=True,
    seed=42
)
save_ptr_data_to_file('data/test_realistic.dat', freq, amp, phase)
print("Realistic PTR test data saved.")