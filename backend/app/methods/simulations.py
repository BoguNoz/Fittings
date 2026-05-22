import numpy as np
from scipy.integrate import quad
from .transfer_matrix import layer_transfer_matrix, interface_matrix

def simulate_single_frequency(omega: float, k2: float, alfa2: float, r32: float, k3: float,
                              config, anisotropy: float = 1.0):
    """
    Zwraca zespolony sygnał TR (model Henkela 3D) dla jednej częstotliwości.
    Uwzględnia dzielenie przez sqrt(omega), które jest częścią definicji sygnału.
    """
    d = config.d_pump
    upper = 35.0 / d          # górna granica całkowania Hankela

    def integrand(lam):
        return _theta_contribution(lam, omega, k2, alfa2, r32, k3, config, anisotropy)

    # Całkowanie numeryczne (osobno część rzeczywista i urojona dla bezpieczeństwa)
    real, _ = quad(lambda lam: np.real(integrand(lam)), 0, upper, limit=150, epsabs=1e-9, epsrel=1e-6)
    imag, _ = quad(lambda lam: np.imag(integrand(lam)), 0, upper, limit=150, epsabs=1e-9, epsrel=1e-6)

    theta = (real + 1j * imag) * (-config.Q / (2 * np.pi))   # Eq. (4)
    signal = theta / np.sqrt(omega)                          # normalizacja / sqrt(omega)
    return signal

def _theta_contribution(lam: float, omega: float, k2: float, alfa2: float, r32: float,
                        k3: float, config, anisotropy: float = 1.0):
    if lam < 1e-30:
        return 0j

    # Macierze przejścia dla trzech warstw
    M1 = layer_transfer_matrix(config.k1, config.alfa1, config.l1, lam, omega)
    M2 = layer_transfer_matrix(k2, alfa2, config.l2, lam, omega,
                               k_in=k2 * anisotropy)   # anizotropia uwzględniona
    M3 = layer_transfer_matrix(k3, config.alfa3, 1e-3, lam, omega)

    # Złożenie macierzy z oporami międzywarstwowymi
    M = (M1 @ interface_matrix(config.r21) @
         M2 @ interface_matrix(r32) @
         M3)

    C, D = M[0, 0], M[0, 1]
    theta = -D / C if abs(C) > 1e-200 else 0j

    # Czynnik Gaussa od rozkładu wiązki pompującej
    gauss = np.exp(-(lam * config.d_pump)**2 / 4.0)
    return theta * gauss * lam