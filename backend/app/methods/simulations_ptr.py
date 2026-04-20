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

    Parameters
    ----------
    frequency_vector : np.ndarray
        Vector of frequencies in Hz.
    k2 : float
        Thermal conductivity of the main ZnO film (Layer 2) [W/(m·K)].
    alfa2 : float
        Thermal diffusivity of the main ZnO film (Layer 2) [m²/s].
    r32 : float
        Thermal boundary resistance (TBR) between ZnO film and substrate (Layer 2–3) [m²·K/W].
    k3 : float
        Thermal conductivity of the substrate (Layer 3) [W/(m·K)].

    # Fixed/default parameters (can be overridden)
    k1 : float, default=21.0
        Thermal conductivity of the top layer (Layer 1) [W/(m·K)].
    l1 : float, default=80e-9
        Thickness of the top surface layer (Layer 1) [m].
    l2 : float, default=469e-9
        Thickness of the main ZnO film (Layer 2) [m].
    alfa1 : float, default=8.9e-6
        Thermal diffusivity of the top layer (Layer 1) [m²/s].
    alfa3 : float, default=6.0e-6
        Thermal diffusivity of the substrate (Layer 3) [m²/s].
    r21 : float, default=2.8e-8
        Thermal boundary resistance between Layer 1 and Layer 2 [m²·K/W].

    Returns
    -------
    amplitude : np.ndarray
        Amplitude of the PTR signal (not normalized).
    y_complex : np.ndarray
        Complex PTR response (temperature oscillation / √ω).
    """
    omega = 2 * np.pi * frequency_vector

    # Thermal wave numbers (complex)
    s1 = np.sqrt(1j * omega / alfa1)
    s2 = np.sqrt(1j * omega / alfa2)
    s3 = np.sqrt(1j * omega / alfa3)

    # Reflection coefficients at interfaces
    r12 = (k1 * s1) / (k2 * s2)

    # Tangent of thermal wave phase shift in each layer
    t1 = np.tan(s1 * l1)
    t2 = np.tan(s2 * l2)

    # Precompute repeated terms
    p1 = k1 * s1 * t1
    p2 = k2 * s2 * t2
    cross = r21 * p1 * p2

    # Main coefficients of the thermal wave model
    A = 1 + r21 * p1 + r12 * t1 * t2 + r32 * (p1 + p2 - cross)
    B = -t1 / (k1 * s1) - t2 / (k2 * s2) - r21 - r32 * (1 + r21 * t1 * t2 - r21 * k2 * s2 * t2)
    G = -p1 - p2 - cross
    D = 1 + r21 * (t1 * t2 + k2 * s2 * t2)

    # Complex temperature oscillation at the surface
    ypt3m = -(G - k3 * s3 * A) / (D - k3 * s3 * B)

    # PTR signal is proportional to temperature oscillation divided by sqrt(omega)
    y_complex = ypt3m / np.sqrt(omega)

    amplitude = np.abs(y_complex)

    return amplitude, y_complex