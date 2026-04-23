import numpy as np
from app.methods.simulations_ptr import simulations_ptr


def ptr_residual(
        p: np.ndarray,
        frequency_vector: np.ndarray,
        exp_amp: np.ndarray,
        exp_phase: np.ndarray,
        phase_units: str,
        **phys_params
) -> np.ndarray:
    """
    Residual function for least_squares optimization.
    p: [log10(k2), log10(alfa2), log10(r32), log10(k3), phi0_deg]
    """
    # Unpack parameters (log10 scale for stability)
    k2 = 10 ** p[0]
    alfa2 = 10 ** p[1]
    r32 = 10 ** p[2]
    k3 = 10 ** p[3]
    phi0_deg = p[4]

    # Compute model response with provided physical constants
    _, y_complex = simulations_ptr(
        frequency_vector, k2, alfa2, r32, k3, **phys_params
    )

    # Normalize model to the first frequency point and apply fitted phase offset
    phi0_rad = np.deg2rad(phi0_deg)
    y_normalized = (y_complex / y_complex[0]) * np.exp(1j * phi0_rad)

    # Prepare experimental complex signal
    if phase_units.lower() == "deg":
        exp_phase_rad = np.deg2rad(exp_phase)
    else:
        exp_phase_rad = exp_phase

    # Normalize experimental amplitude to match model normalization
    exp_normalized = (exp_amp / exp_amp[0]) * np.exp(1j * exp_phase_rad)

    # Frequency-based weighting (emphasizes high-frequency data points)
    weight = (frequency_vector / frequency_vector.max()) ** 0.8

    # Calculate complex relative difference
    diff = (y_normalized - exp_normalized) / np.maximum(np.abs(exp_normalized), 1e-12)
    diff = diff * weight

    # Return concatenated real and imaginary parts for the optimizer
    return np.concatenate([np.real(diff), np.imag(diff)])