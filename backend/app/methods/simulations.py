import numpy as np
from scipy.integrate import quad
from app.methods.transfer_matrix import layer_transfer_matrix, interface_matrix

def simulations_ptr_hankel(freq_khz: np.ndarray, k2: float, alfa2: float, r32: float, k3: float,
                           anisotropy: float = 1.0, **kwargs):
    """Stabilna wersja z ograniczeniami i fallbackiem."""
    params = dict(kwargs)
    params.update({
        'k2': max(0.01, float(k2)),
        'alpha2': max(1e-9, float(alfa2)),
        'R32': max(1e-10, float(r32)),
        'anisotropy': max(1.0, float(anisotropy)),
        'k3': max(0.1, float(k3))
    })

    # Uzupełnij brakujące klucze
    defaults = {
        'k1': 200.0, 'l1': 50e-9, 'alfa1': 8.2e-5,
        'l2': 240e-9, 'alfa3': 0.5e-7, 'r21': 1e-8,
        'd_pump': 2.42e-6, 'Q': 1.0
    }
    for k, v in defaults.items():
        if k not in params:
            params[k] = v

    omega_vec = 2 * np.pi * np.asarray(freq_khz) * 1000
    y_complex = np.full(len(omega_vec), 1e-8 + 0j, dtype=complex)

    for i, omega in enumerate(omega_vec):
        try:
            y_complex[i] = _simulate_single_freq(omega, params)
        except Exception:
            pass  # fallback do małej wartości

    return np.abs(y_complex), y_complex


def _simulate_single_freq(omega: float, params: dict):
    d = float(params['d_pump'])
    upper = 30.0 / d   # zmniejszone dla stabilności

    def real_int(lam):
        val = _theta_contribution(lam, omega, params)
        return float(np.real(val))

    def imag_int(lam):
        val = _theta_contribution(lam, omega, params)
        return float(np.imag(val))

    real, _ = quad(real_int, 0, upper, limit=100, epsabs=1e-8, epsrel=1e-5)
    imag, _ = quad(imag_int, 0, upper, limit=100, epsabs=1e-8, epsrel=1e-5)

    theta = (real + 1j * imag) * (-params['Q'] / (2 * np.pi))
    return theta / np.sqrt(omega + 1e-30)


def _theta_contribution(lam: float, omega: float, params: dict):
    if lam < 1e-30:
        return 0j
    try:
        M1 = layer_transfer_matrix(params['k1'], params['alfa1'], params['l1'], lam, omega)
        M2 = layer_transfer_matrix(params['k2'], params['alpha2'], params['l2'], lam, omega,
                                   k_in=params['k2'] * params['anisotropy'])
        M3 = layer_transfer_matrix(params['k3'], params['alfa3'], 1e-3, lam, omega)

        M = M1 @ interface_matrix(params['r21']) @ M2 @ interface_matrix(params['R32']) @ M3

        C = M[0, 0]
        D = M[0, 1]
        theta = -D / C if abs(C) > 1e-200 else 0j

        gauss = np.exp(-(lam * params['d_pump'])**2 / 4.0)
        return theta * gauss * lam
    except:
        return 0j