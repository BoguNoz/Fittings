import numpy as np

def simulations_ptr(
        frequency_vector: np.ndarray,
        k2: float,
        alfa2: float,
        r32: float,
        k3: float,
        *,
        k1: float = 21.0,
        l1: float = 80e-9,
        l2: float = 469e-9,
        alfa1: float = 8.9e-6,
        alfa3: float = 6.0e-6,
        r21: float = 2.8e-8,
        **kwargs  # Captures extra config arguments like 'alfa2_conf' to prevent TypeErrors
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the complex PTR (Photothermal Radiometry) response for a 3-layer system.

    Args:
        frequency_vector: Array of modulation frequencies [Hz].
        k2: Thermal conductivity of the second layer [W/mK].
        alfa2: Thermal diffusivity of the second layer [m^2/s].
        r32: Thermal boundary resistance between layer 3 and 2 [m^2K/W].
        k3: Thermal conductivity of the substrate (layer 3) [W/mK].
        k1, l1, l2, alfa1, alfa3, r21: Fixed physical constants.
        **kwargs: Catch-all for additional configuration parameters.

    Returns:
        tuple: (amplitude array, complex response array).
    """
    omega = 2 * np.pi * frequency_vector

    # Thermal wave numbers (complex sigma) for each layer
    s1 = np.sqrt(1j * omega / alfa1)
    s2 = np.sqrt(1j * omega / alfa2)
    s3 = np.sqrt(1j * omega / alfa3)

    # Thermal effusivity ratios and reflection coefficients
    ro12 = (k1 * s1) / (k2 * s2)
    ro21 = (k2 * s2) / (k1 * s1)

    # Phase shifts within layers (complex tangents)
    t1 = np.tan(s1 * l1)
    t2 = np.tan(s2 * l2)

    # Matrix coefficients for the 3-layer thermal wave propagation
    A = (1 + r21 * k1 * s1 * t1 + ro12 * t1 * t2 +
         r32 * (k1 * s1 * t1 + k2 * s2 * t2 - r21 * k1 * s1 * k2 * s2 * t1 * t2))

    B = (-t1 / (k1 * s1) - t2 / (k2 * s2) - r21 -
         r32 * (1 + ro21 * t1 * t2 - r21 * k2 * s2 * t2))

    G = -k1 * s1 * t1 - k2 * s2 * t2 - r21 * k1 * s1 * k2 * s2 * t1 * t2

    D = 1 + ro21 * t1 * t2 + r21 * k2 * s2 * t2

    # Surface temperature oscillation solution
    ypt3m = -(G - k3 * s3 * A) / (D - k3 * s3 * B)

    # Frequency-dependent normalization for the radiometry signal
    y_complex = ypt3m / np.sqrt(omega)
    amplitude = np.abs(y_complex)

    return amplitude, y_complex