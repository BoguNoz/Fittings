import numpy as np
from scipy.integrate import quad
from app.methods.transfer_matrix import layer_transfer_matrix, interface_matrix

def simulations_ptr_hankel(freq_khz: np.ndarray, k2: float, alfa2: float, r32: float, k3: float,
                           anisotropy: float = 1.0, **kwargs):
    """Main simulation function - returns (amplitude, complex_signal)"""
    params = {**kwargs}
    params.update({
        'k2': k2, 'alpha2': alfa2, 'R32': r32,
        'anisotropy': anisotropy, 'k3': k3
    })

    omega_vec = 2 * np.pi * freq_khz * 1000
    y_complex = np.zeros(len(omega_vec), dtype=complex)

    for i, omega in enumerate(omega_vec):
        y_complex[i] = _simulate_single_freq(omega, params)

    return np.abs(y_complex), y_complex


def _simulate_single_freq(omega: float, params: dict):
    d = params.get('d_pump', 2.42e-6)
    upper = 40.0 / d

    def integrand_real(lam):
        return np.real(_theta_contribution(lam, omega, params))
    def integrand_imag(lam):
        return np.imag(_theta_contribution(lam, omega, params))

    real, _ = quad(integrand_real, 0, upper, limit=200, epsabs=1e-9, epsrel=1e-6)
    imag, _ = quad(integrand_imag, 0, upper, limit=200, epsabs=1e-9, epsrel=1e-6)

    theta = (real + 1j * imag) * (-params.get('Q', 1.0) / (2 * np.pi))
    return theta / np.sqrt(omega + 1e-30)


def _theta_contribution(lam: float, omega: float, params: dict):
    if lam < 1e-30:
        return 0j

    M1 = layer_transfer_matrix(params['k1'], params['alfa1'], params['l1'], lam, omega)
    M2 = layer_transfer_matrix(params['k2'], params['alpha2'], params['l2'], lam, omega,
                               k_in=params['k2'] * params['anisotropy'])
    M3 = layer_transfer_matrix(params['k3'], params['alfa3'], 1e-3, lam, omega)

    M = M1 @ interface_matrix(params['r21']) @ M2 @ interface_matrix(params['R32']) @ M3

    C, D = M[0,0], M[0,1]
    theta = -D / C if abs(C) > 1e-280 else 0j
    gauss = np.exp(-(lam * params['d_pump'])**2 / 4.0)
    return theta * gauss * lam