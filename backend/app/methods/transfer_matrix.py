import numpy as np


def thermal_wavevector(lam: float, omega: float, alfa: float, anisotropy: float = 1.0) -> complex:
    """
    Oblicza zespoloną liczbę falową fali cieplnej dla warstwy.

    Parametry:
    lam - zmienna transformaty Hankela (radialna liczba falowa) [1/m]
    omega - pulsacja [rad/s]
    alfa - dyfuzyjność cieplna [m^2/s]
    anisotropy - stosunek przewodności radialnej do pionowej (K_r/K_z) dla warstwy anizotropowej
    """
    return np.sqrt(anisotropy * lam ** 2 + 1j * omega / alfa)


def layer_transfer_matrix(sigma: complex, k: float, d: float) -> np.ndarray:
    """
    Macierz transferowa dla jednorodnej warstwy o grubości d.
    Łączy wektor [T; q] na górze i dole warstwy.
    Uwaga: q = -k ∂T/∂z (strumień ciepła w kierunku +z).
    """
    ch = np.cosh(sigma * d)
    sh = np.sinh(sigma * d)
    # Standardowa macierz:
    # [ T_top ]   [  ch     -sh/(k*sigma) ] [ T_bottom ]
    # [ q_top ] = [ -k*sigma*sh     ch    ] [ q_bottom ]
    M = np.array([
        [ch, -sh / (k * sigma)],
        [-k * sigma * sh, ch]
    ], dtype=complex)
    return M


def interface_matrix(R: float) -> np.ndarray:
    """Macierz dla interfejsu z oporem termicznym R."""
    return np.array([[1, R], [0, 1]], dtype=float)