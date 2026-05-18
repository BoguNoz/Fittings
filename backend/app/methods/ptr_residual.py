import numpy as np
from app.methods.simulations_ptr_hankel import simulations_ptr_hankel


def ptr_residual(
        p: np.ndarray,
        frequency_vector: np.ndarray,
        exp_amp: np.ndarray,
        exp_phase_deg: np.ndarray,
        **phys_params
) -> np.ndarray:
    """
    Weighted complex residual for PTR fitting with Hankel model.
    Parameter vector p (log10 scale):
        [k2, anisotropy, r32, k3, phi0_deg]
    alfa2 is computed from k2 and volumetric heat capacity (rhoc).
    """
    # Unpack and convert from log10
    k2 = 10 ** p[0]
    anisotropy = 10 ** p[1]
    r32 = 10 ** p[2]
    k3 = 10 ** p[3]
    phi0_rad = np.deg2rad(p[4])

    # Link alfa2 to k2 using volumetric heat capacity
    rhoc = phys_params.get('rhoc', 2.5e6)       # J/m³·K
    alfa2 = k2 / rhoc

    # Fixed physical parameters passed through phys_params
    model_kwargs = {key: phys_params[key] for key in
                    ['k1', 'l1', 'l2', 'alfa1', 'alfa3', 'r21', 'd_pump', 'Q']
                    if key in phys_params}

    # Generate complex model response (raw, not normalized)
    _, y_complex = simulations_ptr_hankel(
        frequency_vector, k2, alfa2, r32, k3,
        anisotropy=anisotropy,
        **model_kwargs
    )

    # Normalize both model and experiment to the first point,
    # then apply global phase offset.
    y_norm = (y_complex / y_complex[0]) * np.exp(1j * phi0_rad)

    exp_phase_rad = np.deg2rad(exp_phase_deg)
    exp_complex_raw = exp_amp * np.exp(1j * exp_phase_rad)
    e_norm = (exp_complex_raw / exp_complex_raw[0]) * np.exp(1j * phi0_rad)

    # Frequency‑dependent weighting (more weight to mid‑high frequencies)
    w_exp = phys_params.get('weight_exponent', 0.5)
    weight = (frequency_vector / frequency_vector.max()) ** w_exp

    # Complex relative difference
    diff = (y_norm - e_norm) / np.maximum(np.abs(e_norm), 1e-12)
    diff *= weight.reshape(-1, 1) if diff.ndim > 1 else weight

    # Extra weight on imaginary part (phase information)
    phase_weight = phys_params.get('phase_weight', 1.5)
    return np.concatenate([np.real(diff), np.imag(diff) * phase_weight])