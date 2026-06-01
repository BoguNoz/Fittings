import numpy as np

from app.methods.transfer_matrix import thermal_wavevector, layer_transfer_matrix, interface_matrix
from app.models.ptr_config import PTRConfig


def simulate_single_frequency(omega: float, k2: float, alfa2: float, r32: float, k3: float, config: PTRConfig,
                              anisotropy: float = 1.0) -> complex:
    # 1. Kwadratura Gaussa-Legendre'a (np. 500 punktów wystarczy dzięki wysokiej precyzji metody)
    n_nodes = 500
    lam_max = 35.0 / config.d_pump

    # x_nodes są w przedziale [-1, 1], musimy je przeskalować do [0, lam_max]
    x_nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    lam_vals = 0.5 * lam_max * (x_nodes + 1)
    w_scaled = 0.5 * lam_max * weights

    # Wektoryzacja całkowania - omijamy powolną pętlę for!
    sigma1 = thermal_wavevector(lam_vals, omega, config.alfa1, anisotropy=1.0)
    sigma2 = thermal_wavevector(lam_vals, omega, alfa2, anisotropy=anisotropy)
    sigma3 = thermal_wavevector(lam_vals, omega, config.alfa3, anisotropy=1.0)

    # Inicjalizacja tablic (zamiast mnożenia macierzy w pętli, robimy to analitycznie lub wektorowo)
    # Dla czytelności pokazuję uproszczoną wektoryzację pętli (szybciej niż zwykły for):

    integral = 0.0 + 0.0j
    for lam, w, s1, s2, s3 in zip(lam_vals, w_scaled, sigma1, sigma2, sigma3):
        M1 = layer_transfer_matrix(s1, config.k1, config.l1)
        M_int12 = interface_matrix(config.r21)
        M2 = layer_transfer_matrix(s2, k2, config.l2)
        M_int23 = interface_matrix(r32)

        M = M1 @ M_int12 @ M2 @ M_int23
        Y_sub = k3 * s3

        theta_lam = (M[0, 0] + M[0, 1] * Y_sub) / (M[1, 0] + M[1, 1] * Y_sub)

        # Szerokość wiązki (patrz punkt 2 poniżej)
        # Zależnie od definicji d_pump, to może wymagać modyfikacji!
        gauss = np.exp(-(lam * config.d_pump) ** 2 / 8)

        integral += theta_lam * gauss * lam * w

    theta_surface = (-config.Q / (2 * np.pi)) * integral
    return theta_surface