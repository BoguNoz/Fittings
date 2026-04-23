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
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the complex photothermal radiometry (PTR) response for a three-layer thermal wave model.

    Layers:
        Layer 1: Thin top surface layer (e.g. damaged/implantation layer)
        Layer 2: Main ZnO film (the layer of interest)
        Layer 3: Glass substrate
    """
    omega = 2 * np.pi * frequency_vector

    # Thermal wave numbers (complex)
    s1 = np.sqrt(1j * omega / alfa1)
    s2 = np.sqrt(1j * omega / alfa2)
    s3 = np.sqrt(1j * omega / alfa3)

    # Reflection coefficients / Thermal effusivity ratios as defined in the original MATLAB script
    ro12 = (k1 * s1) / (k2 * s2)
    ro21 = (k2 * s2) / (k1 * s1)

    # Tangent of thermal wave phase shift in each layer
    t1 = np.tan(s1 * l1)
    t2 = np.tan(s2 * l2)

    # Coefficients for the 3-layer system matrix (Matching MATLAB implementation exactly)
    # alpha (A)
    A = (1 + r21 * k1 * s1 * t1 + ro12 * t1 * t2 +
         r32 * (k1 * s1 * t1 + k2 * s2 * t2 - r21 * k1 * s1 * k2 * s2 * t1 * t2))

    # beta (B)
    B = (-t1 / (k1 * s1) - t2 / (k2 * s2) - r21 -
         r32 * (1 + ro21 * t1 * t2 - r21 * k2 * s2 * t2))

    # gama (G)
    G = -k1 * s1 * t1 - k2 * s2 * t2 - r21 * k1 * s1 * k2 * s2 * t1 * t2

    # delta (D)
    D = 1 + ro21 * t1 * t2 + r21 * k2 * s2 * t2

    # Complex temperature oscillation at the surface
    ypt3m = -(G - k3 * s3 * A) / (D - k3 * s3 * B)

    # PTR signal is proportional to temperature oscillation divided by sqrt(omega)
    # The absolute phase of the system includes this sqrt(omega) term
    y_complex = ypt3m / np.sqrt(omega)
    amplitude = np.abs(y_complex)

    return amplitude, y_complex