import numpy as np
from scipy.integrate import quad
from app.methods.transfer_matrix import thermal_wavevector, layer_transfer_matrix, interface_matrix
from app.models.ptr_config import PTRConfig


def simulate_single_frequency(omega: float, k2: float, alfa2: float, r32: float, k3: float, config: PTRConfig,
                              anisotropy: float = 1.0) -> complex:
    # 1. Definiujemy wewnętrzną funkcję podcałkową dla pojedynczego lambda (skalara)
    def integrand_core(lam):
        sigma1 = thermal_wavevector(lam, omega, config.alfa1, anisotropy=1.0)
        sigma2 = thermal_wavevector(lam, omega, alfa2, anisotropy=anisotropy)
        sigma3 = thermal_wavevector(lam, omega, config.alfa3, anisotropy=1.0)

        # Wykorzystujemy Twoje funkcje importowane z transfer_matrix
        M1 = layer_transfer_matrix(sigma1, config.k1, config.l1)
        M_int12 = interface_matrix(config.r21)
        M2 = layer_transfer_matrix(sigma2, k2, config.l2)
        M_int23 = interface_matrix(r32)

        M = M1 @ M_int12 @ M2 @ M_int23
        Y_sub = k3 * sigma3

        theta_lam = (M[0, 0] + M[0, 1] * Y_sub) / (M[1, 0] + M[1, 1] * Y_sub)

        # Szerokość wiązki (Beam Spot) – poprawiony dzielnik na 8 zgodnie z transformatą Hankela
        gauss = np.exp(-(lam * config.d_pump) ** 2 / 8)

        return theta_lam * gauss * lam

    # 2. Zakres całkowania (tam, gdzie funkcja Gaussa bezpiecznie wygasza do zera)
    lam_max = 35.0 / config.d_pump

    # 3. Adaptacyjne całkowanie numeryczne (Odpowiednik MATLAB quadgk)
    # Ponieważ scipy.integrate.quad nie obsługuje bezpośrednio liczb zespolonych,
    # rozbijamy całkę na część rzeczywistą (real) i urojoną (imag).
    res_real, _ = quad(lambda l: np.real(integrand_core(l)), 0, lam_max, epsabs=1e-10, epsrel=1e-10, limit=100)
    res_imag, _ = quad(lambda l: np.imag(integrand_core(l)), 0, lam_max, epsabs=1e-10, epsrel=1e-10, limit=100)

    integral = res_real + 1j * res_imag

    # Ostateczna temperatura powierzchniowa
    theta_surface = (-config.Q / (2 * np.pi)) * integral
    return theta_surface