import numpy as np

from app.methods.simulations_ptr import simulations_ptr


def ptr_residual(
        p: np.ndarray, # Optimization parameters in log10 scale:
        frequency_vector: np.ndarray, #  Frequency vector in Hz (independent variable)
        exp_amp: np.ndarray, #  Experimental amplitude values
        exp_phase: np.ndarray, # Experimental phase values
        phase_units: str = "deg", # Units of the experimental phase: 'deg' or 'rad'
) -> np.ndarray:
    """
    Residual function used for least_squares optimization of PTR data.

    Returns
    -------
    np.ndarray
        Stacked vector of weighted real and imaginary residuals.
    """

    # Unpack parameters from log10 scale
    k2 = 10 ** p[0]
    alfa2 = 10 ** p[1]
    r32 = 10 ** p[2]
    k3 = 10 ** p[3]
    phi0_deg = p[4]

    # Compute complex model response
    _, y_complex = simulations_ptr(
        frequency_vector, k2, alfa2, r32, k3
    )

    # Normalize model to first point and apply phase offset
    phi0_rad = phi0_deg * np.pi / 180.0
    y_normalized = (y_complex / y_complex[0]) * np.exp(1j * phi0_rad)

    # Prepare experimental complex signal
    if phase_units.lower() == "deg":
        exp_phase_rad = exp_phase * np.pi / 180.0
    else:
        exp_phase_rad = exp_phase

    exp_normalized = (exp_amp / exp_amp[0]) * np.exp(1j * exp_phase_rad)

    # Weighting factor - gives more importance to higher frequencies
    weight = (frequency_vector / frequency_vector.max()) ** 0.8

    # Relative complex difference
    diff = (y_normalized - exp_normalized) / np.maximum(np.abs(exp_normalized), 1e-12)
    diff = diff * weight

    # Return stacked real + imaginary parts for least_squares
    return np.concatenate([np.real(diff), np.imag(diff)])