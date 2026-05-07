import numpy as np
from app.methods.simulations_ptr import simulations_ptr

def ptr_residual(
        p: np.ndarray,
        frequency_vector: np.ndarray,
        exp_amp: np.ndarray,
        exp_phase_deg: np.ndarray,
        **phys_params
) -> np.ndarray:
    """
    Computes the weighted complex residual between the model and experimental data.
    Input phase is expected in DEGREES and converted to RADIANS for physics calculations.
    """
    # Unpack parameters from log10 space for numerical stability
    k2, alfa2, r32, k3 = 10 ** p[0], 10 ** p[1], 10 ** p[2], 10 ** p[3]
    phi0_rad = np.deg2rad(p[4])

    # 1. Generate model response using current optimization parameters
    _, y_complex = simulations_ptr(frequency_vector, k2, alfa2, r32, k3, **phys_params)

    # 2. Normalize model to its first frequency point and apply fitted phase offset
    y_norm = (y_complex / y_complex[0]) * np.exp(1j * phi0_rad)

    # 3. Convert experimental data from degrees to radians
    exp_phase_rad = np.deg2rad(exp_phase_deg)
    exp_complex_raw = exp_amp * np.exp(1j * exp_phase_rad)

    # 4. Normalize experimental data relative to its first frequency point
    e_norm = (exp_complex_raw / exp_complex_raw[0]) * np.exp(1j * phi0_rad)

    # 5. Frequency weighting to emphasize high-frequency characteristics
    weight = (frequency_vector / frequency_vector.max()) ** 0.8

    # 6. Compute complex relative difference
    diff = (y_norm - e_norm) / np.maximum(np.abs(e_norm), 1e-12)
    diff *= weight

    # Return stacked real and imaginary components.
    # Imaginary part is weighted by 2.0 to prioritize phase fitting accuracy.
    return np.concatenate([np.real(diff), np.imag(diff) * 2.0])