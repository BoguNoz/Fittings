import numpy as np

def simulations_ptr(
        frequency_vector: np.ndarray,
        k2: float,
        alfa2: float,
        r32: float,
        k3: float,
        *,
        k1: float = 200.0,          # Aluminium (typical)
        l1: float = 50e-9,          # 50 nm Al transducer
        l2: float = 240e-9,         # PEDOT:PSS thickness (adjust to sample)
        alfa1: float = 8.2e-5,      # Al thermal diffusivity
        alfa3: float = 0.5e-6,      # Glass substrate (corrected from 0.5e-7)
        r21: float = 1.0e-8,        # Al/PEDOT interface resistance
        **kwargs                    # Absorbs extra config keys (e.g., 'phase_weight')
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the complex PTR response for a 3-layer system (metal / thin film / substrate).
    Returns (amplitude, complex_response).
    """
    omega = 2 * np.pi * frequency_vector

    # Thermal wave numbers
    s1 = np.sqrt(1j * omega / alfa1)
    s2 = np.sqrt(1j * omega / alfa2)
    s3 = np.sqrt(1j * omega / alfa3)

    # Effusivity ratios
    ro12 = (k1 * s1) / (k2 * s2)
    ro21 = (k2 * s2) / (k1 * s1)

    # Tangents of complex phase shifts
    t1 = np.tan(s1 * l1)
    t2 = np.tan(s2 * l2)

    # Matrix coefficients (standard multilayer thermal wave solution)
    A = (1 + r21 * k1 * s1 * t1 + ro12 * t1 * t2 +
         r32 * (k1 * s1 * t1 + k2 * s2 * t2 - r21 * k1 * s1 * k2 * s2 * t1 * t2))

    B = (-t1 / (k1 * s1) - t2 / (k2 * s2) - r21 -
         r32 * (1 + ro21 * t1 * t2 - r21 * k2 * s2 * t2))

    G = -k1 * s1 * t1 - k2 * s2 * t2 - r21 * k1 * s1 * k2 * s2 * t1 * t2

    D = 1 + ro21 * t1 * t2 + r21 * k2 * s2 * t2

    # Surface temperature solution
    ypt3m = -(G - k3 * s3 * A) / (D - k3 * s3 * B)

    # Radiometry signal includes 1/sqrt(ω) factor
    y_complex = ypt3m / np.sqrt(omega)
    amplitude = np.abs(y_complex)

    return amplitude, y_complex