import numpy as np


def thermal_wavevector(lam, omega, alfa, anisotropy=1.0):
    return np.sqrt(anisotropy * (lam**2 + 1j * omega / alfa))

def layer_transfer_matrix(sigma, k, d):
    ch, sh = np.cosh(sigma * d), np.sinh(sigma * d)
    return np.array([[ch, -sh / (k * sigma)], [-k * sigma * sh, ch]], dtype=complex)

def interface_matrix(R):
    return np.array([[1, R], [0, 1]], dtype=complex)