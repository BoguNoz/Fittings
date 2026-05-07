import numpy as np

from app.methods.simulations_ptr import simulations_ptr
from app.methods.simulations_ptr_hankel import simulations_ptr_hankel


def ptr_residual(
        p: np.ndarray,
        frequency_vector: np.ndarray,
        exp_amp: np.ndarray,
        exp_phase: np.ndarray,
        phase_units: str,
        use_hankel: bool = False,
        **phys_params
) -> np.ndarray:
    """
    Computes the weighted complex residual between model and experiment.
    Used by the least_squares optimizer.
    """
    # Unpack parameters from log10 space
    k2 = 10 ** p[0]
    alfa2 = 10 ** p[1]
    r32 = 10 ** p[2]
    k3 = 10 ** p[3]
    phi0_rad = np.deg2rad(p[4])

    # Generate complex model response
    if use_hankel:
        _, y_complex = simulations_ptr_hankel(frequency_vector, k2, alfa2, r32, k3, **phys_params)
    else:
        _, y_complex = simulations_ptr(frequency_vector, k2, alfa2, r32, k3, **phys_params)

    # Normalize model and apply phase correction
    y_norm = (y_complex / y_complex[0]) * np.exp(1j * phi0_rad)

    # Convert experimental data to complex form
    exp_phase_rad = np.deg2rad(exp_phase) if phase_units.lower() == "deg" else exp_phase
    exp_complex_raw = exp_amp * np.exp(1j * exp_phase_rad)

    # Normalize experiment the same way as the model
    e_norm = (exp_complex_raw / exp_complex_raw[0]) * np.exp(1j * phi0_rad)

    # Weighting - emphasize higher frequencies (common in thermal wave analysis)
    weight = (frequency_vector / frequency_vector.max()) ** 0.8

    # Complex relative error
    diff = (y_norm - e_norm) / np.maximum(np.abs(e_norm), 1e-12)
    diff = diff * weight

    # Return stacked real and imaginary parts (with higher weight on imaginary for phase)
    return np.concatenate([np.real(diff), np.imag(diff) * 5.0])