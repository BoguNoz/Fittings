import numpy as np

from app.methods.simulations_ptr import simulations_ptr


def ptr_residual(
    p: np.ndarray,
    frequency_vector: np.ndarray,
    exp_amp: np.ndarray,
    exp_phase: np.ndarray,
    phase_units: str = "deg",
) -> np.ndarray:
    """
    Residual function for least_squares optimization of PTR amplitude and phase data.

    Parameters are optimized in log10 scale for better numerical stability.

    Parameters
    ----------
    p : np.ndarray
        Optimization vector: [log10(k2), log10(alfa2), log10(r32), log10(k3), phi0_deg]
    frequency_vector : np.ndarray
        Frequencies in Hz.
    exp_amp : np.ndarray
        Experimental amplitude values (normalized or absolute).
    exp_phase : np.ndarray
        Experimental phase values (in degrees or radians).
    phase_units : str, default="deg"
        Units of exp_phase: 'deg' or 'rad'.

    Returns
    -------
    np.ndarray
        Stacked real and imaginary weighted residuals (shape: 2 * N).
    """
    # Unpack parameters (log10 scale except phase offset)
    k2 = 10 ** p[0]
    alfa2 = 10 ** p[1]
    r32 = 10 ** p[2]
    k3 = 10 ** p[3]
    phi0_deg = p[4]

    # Compute model response
    _, y_complex = simulations_ptr(
        frequency_vector, k2, alfa2, r32, k3
    )

    # Normalize model to first frequency point and apply phase offset
    phi0_rad = np.deg2rad(phi0_deg)
    y_normalized = (y_complex / y_complex[0]) * np.exp(1j * phi0_rad)

    # Prepare experimental complex signal
    if phase_units.lower() == "deg":
        exp_phase_rad = np.deg2rad(exp_phase)
    else:
        exp_phase_rad = exp_phase

    exp_normalized = (exp_amp / exp_amp[0]) * np.exp(1j * exp_phase_rad)

    # Weighting: higher frequencies get more importance (0.8 is a good compromise)
    weight = (frequency_vector / frequency_vector.max()) ** 0.8

    # Relative complex difference (more robust than absolute)
    diff = (y_normalized - exp_normalized) / np.maximum(np.abs(exp_normalized), 1e-12)
    diff = diff * weight

    # Return stacked real + imaginary parts for least_squares
    return np.concatenate([np.real(diff), np.imag(diff)])