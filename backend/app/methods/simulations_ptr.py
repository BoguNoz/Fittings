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
    Computes the complex PTR response based on the 3-layer MATLAB model.
    Includes the critical sqrt(1j * omega) factor for correct phase behavior.
    """
    omega = 2 * np.pi * frequency_vector

    # Thermal wave numbers (complex sigma)
    s1 = np.sqrt(1j * omega / alfa1)
    s2 = np.sqrt(1j * omega / alfa2)
    s3 = np.sqrt(1j * omega / alfa3)

    # Reflection / ratio coefficients
    ro12 = (k1 * s1) / (k2 * s2)
    ro21 = (k2 * s2) / (k1 * s1)

    # Layer phase shifts (tangents)
    t1 = np.tan(s1 * l1)
    t2 = np.tan(s2 * l2)

    # Matrix coefficients (A, B, G, D) following MATLAB script exactly
    A = (1 + r21 * k1 * s1 * t1 + ro12 * t1 * t2 +
         r32 * (k1 * s1 * t1 + k2 * s2 * t2 - r21 * k1 * s1 * k2 * s2 * t1 * t2))

    B = (-t1 / (k1 * s1) - t2 / (k2 * s2) - r21 -
         r32 * (1 + ro21 * t1 * t2 - r21 * k2 * s2 * t2))

    G = -k1 * s1 * t1 - k2 * s2 * t2 - r21 * k1 * s1 * k2 * s2 * t1 * t2

    D = 1 + ro21 * t1 * t2 + r21 * k2 * s2 * t2

    # Surface temperature oscillation
    ypt3m = -(G - k3 * s3 * A) / (D - k3 * s3 * B)

    # IMPORTANT: The complex response must include the diffusion factor sqrt(j*omega).
    # This ensures the model's phase can rotate correctly.
    y_complex = ypt3m / np.sqrt(1j * omega)
    amplitude = np.abs(y_complex)

    return amplitude, y_complex