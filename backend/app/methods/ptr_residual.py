import numpy as np
from app.methods.simulations_ptr_hankel import simulations_ptr_hankel

def ptr_residual(
        p: np.ndarray,
        frequency_vector: np.ndarray,
        exp_amp: np.ndarray,
        exp_phase_deg: np.ndarray,
        **phys_params
) -> np.ndarray:
    k2 = 10 ** p[0]
    anisotropy = 10 ** p[1]
    r32 = 10 ** p[2]
    k3 = 10 ** p[3]

    rhoc = phys_params.get('rhoc', 2.0e6)
    alfa2 = k2 / rhoc

    model_kwargs = {key: phys_params[key] for key in
                    ['k1', 'l1', 'l2', 'alfa1', 'alfa3', 'r21', 'd_pump', 'Q']
                    if key in phys_params}

    _, y_complex = simulations_ptr_hankel(
        frequency_vector, k2, alfa2, r32, k3,
        anisotropy=anisotropy, **model_kwargs
    )

    phase_rad = np.unwrap(np.deg2rad(exp_phase_deg))
    exp_complex = exp_amp * np.exp(1j * phase_rad)

    G = np.vdot(y_complex, exp_complex) / np.vdot(y_complex, y_complex)
    model_scaled = G * y_complex

    f_min = frequency_vector[0]
    weight = np.sqrt(f_min / frequency_vector)
    denom = np.maximum(np.abs(exp_complex), 1e-12)

    diff = (model_scaled - exp_complex) / denom

    phase_weight = phys_params.get('phase_weight', 1.5)
    return np.concatenate([np.real(diff) * weight,
                           np.imag(diff) * weight * phase_weight])