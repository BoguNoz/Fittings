import numpy as np

def thermal_wavevector(k_perp: float, alpha_perp: float, k_in: float, lam: float, omega: float):
    """Wektor falowy (równanie (12) z artykułu)"""
    return np.sqrt(1j * omega / alpha_perp + (k_in / k_perp) * lam**2 + 0j)

def layer_transfer_matrix(k: float, alpha: float, thickness: float, lam: float, omega: float,
                          k_in=None):
    if thickness < 1e-12:
        return np.eye(2, dtype=complex)
    if k_in is None:
        k_in = k
    q = thermal_wavevector(k, alpha, k_in, lam, omega)
    gamma = q * thickness
    if np.abs(gamma) > 25:
        Z = 1.0 / (k * q)
        return np.array([[1.0, Z], [1.0 / Z, 1.0]], dtype=complex)
    tanh_g = np.tanh(gamma)
    Z = 1.0 / (k * q)
    return np.array([[1.0, Z * tanh_g],
                     [tanh_g / Z, 1.0]], dtype=complex)

def interface_matrix(R: float):
    return np.array([[1.0, R], [0.0, 1.0]], dtype=complex)