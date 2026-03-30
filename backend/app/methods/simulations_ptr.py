import numpy as np


def simulations_ptr (
        frequency_vector: np.ndarray, # Frequency vector in Hz (independent variable)
        k2: float,  # Thermal conductivity of the main ZnO film (Layer 2)
        alfa2: float,  # Thermal diffusivity of the main ZnO film (Layer 2)
        r32: float,  # Thermal boundary resistance between film and substrate (Layer 2–3)

        # Fixed / default parameters:
        k1: float = 21.0, # Thermal conductivity of Layer 1 (top layer)
        k3: float = 3.0, # Thermal conductivity of Layer 3
        l1: float = 80e-9,  # Thickness of top surface layer (Layer 1)
        l2: float = 469e-9,  # Thickness of the main ZnO film (Layer 2)
        alfa1: float = 8.9e-6, # Thermal conductivity of Layer 1
        alfa3: float = 6.0e-6, # Thermal conductivity of Layer 3
        r21: float = 2.8e-8,  # Thermal boundary resistance between Layer 1 and Layer 2
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the complex photothermal radiometry (PTR) response
    for a three-layer thermal model.

    Returns
    -------
    amplitude : np.ndarray
        Amplitude of the PTR signal
    yComplex : np.ndarray
        Complex PTR response (used for phase calculation)
    """
    omega = 2 * np.pi * frequency_vector

    # Thermal wave numbers
    s1 = np.sqrt(1j * omega / alfa1)
    s2 = np.sqrt(1j * omega / alfa2)
    s3 = np.sqrt(1j * omega / alfa3)

    # Reflection coefficients at interfaces
    r12 = k1 * s1 / (k2 * s2)
    r21_calc = k2 * s2 / (k1 * s1)  # renamed to avoid conflict with input parameter r21

    # Tangent terms
    t1 = np.tan(s1 * l1)
    t2 = np.tan(s2 * l2)

    # Precompute repeated expressions for clarity and performance
    p1 = k1 * s1 * t1
    p2 = k2 * s2 * t2
    cross = r21 * p1 * p2

    # Main model coefficients
    A = 1 + r21 * p1 + r12 * t1 * t2 + r32 * (p1 + p2 - cross)
    B = -t1 / (k1 * s1) - t2 / (k2 * s2) - r21 - r32 * (1 + r21 * t1 * t2 - r21 * k2 * s2 * t2)
    G = -p1 - p2 - cross
    D = 1 + r21 * (t1 * t2 + k2 * s2 * t2)

    # Complex temperature oscillation at the surface
    ypt3m = -(G - k3 * s3 * A) / (D - k3 * s3 * B)

    # PTR signal is proportional to temperature / sqrt(omega)
    yComplex = ypt3m / np.sqrt(omega)

    amplitude = np.abs(yComplex)

    return amplitude, yComplex