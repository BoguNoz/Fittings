import numpy as np

from app.methods.simulations_ptr import simulations_ptr
from app.methods.simulations_ptr_hankel import simulations_ptr_hankel


def ptr_residual(
        p: np.ndarray,
        frequency_vector: np.ndarray,
        exp_amp: np.ndarray,
        exp_phase: np.ndarray,
        phase_units: str,
        use_hankel: bool = True,
        **phys_params
) -> np.ndarray:
    """
    Calculates the complex residual between model and experiment.
    Aligns both signals at the first frequency point to ensure phase consistency.
    """
    # Unpack parameters from log10 space
    k2, alfa2, r32, k3 = 10 ** p[0], 10 ** p[1], 10 ** p[2], 10 ** p[3]
    phi0_rad = np.deg2rad(p[4])

    # Get complex model response
    if use_hankel:
        _, y_complex = simulations_ptr_hankel(frequency_vector, k2, alfa2, r32, k3, **phys_params)
    else:
        _, y_complex = simulations_ptr(frequency_vector, k2, alfa2, r32, k3, **phys_params)

    # ALIGNMENT STEP:
    # Normalize model so it starts at (1.0 + 0j) at f[0], then apply phi0 rotation.
    y_norm = (y_complex / y_complex[0]) * np.exp(1j * phi0_rad)

    # Convert experimental phase to radians
    exp_phase_rad = np.deg2rad(exp_phase) if phase_units.lower() == "deg" else exp_phase

    # Create experimental complex signal
    exp_complex_raw = exp_amp * np.exp(1j * exp_phase_rad)

    # ALIGNMENT STEP:
    # Normalize experiment exactly like the model (relative to its own first point).
    # This removes any initial hardware offset and lets phi0 handle the shift.
    e_norm = (exp_complex_raw / exp_complex_raw[0]) * np.exp(1j * phi0_rad)

    # Weighting: emphasize high frequencies (same as MATLAB's 0.8 power)
    weight = (frequency_vector / frequency_vector.max()) ** 0.8

    # Complex relative difference
    diff = (y_norm - e_norm) / np.maximum(np.abs(e_norm), 1e-12)
    diff = diff * weight

    return np.concatenate([np.real(diff), np.imag(diff) * 5.0])