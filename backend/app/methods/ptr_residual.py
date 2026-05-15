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
    Weighted complex residual for PTR fitting.
    Input phase is in DEGREES; internally converted to radians.
    """
    # Unpack parameters from log10 space
    k2, alfa2, r32, k3 = 10 ** p[0], 10 ** p[1], 10 ** p[2], 10 ** p[3]
    phi0_rad = np.deg2rad(p[4])

    # Generate model response
    _, y_complex = simulations_ptr(frequency_vector, k2, alfa2, r32, k3, **phys_params)

    # Normalize model to its first point and apply phase offset
    y_norm = (y_complex / y_complex[0]) * np.exp(1j * phi0_rad)

    # Normalize experimental data (phase already unwrapped!)
    exp_phase_rad = np.deg2rad(exp_phase_deg)
    exp_complex_raw = exp_amp * np.exp(1j * exp_phase_rad)
    e_norm = (exp_complex_raw / exp_complex_raw[0]) * np.exp(1j * phi0_rad)

    # Frequency weighting: lower exponent gives more weight to high frequencies
    # Default 0.5 works better for real data; can be overridden via phys_params['weight_exponent']
    w_exp = phys_params.get('weight_exponent', 0.5)
    weight = (frequency_vector / frequency_vector.max()) ** w_exp

    # Complex relative difference
    diff = (y_norm - e_norm) / np.maximum(np.abs(e_norm), 1e-12)
    diff *= weight

    # Phase weight (default 1.5 for real data, can be changed)
    phase_weight = phys_params.get('phase_weight', 1.5)
    return np.concatenate([np.real(diff), np.imag(diff) * phase_weight])