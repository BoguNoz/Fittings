import numpy as np

from app.methods.transfer_matrix import thermal_wavevector, layer_transfer_matrix, interface_matrix
from app.models.ptr_config import PTRConfig


def simulate_single_frequency(
        omega: float,
        k2: float,
        alfa2: float,
        r32: float,
        k3: float,
        config: PTRConfig,
        anisotropy: float = 1.0
) -> complex:
    """
    Oblicza zespoloną amplitudę temperatury powierzchniowej dla pojedynczej częstotliwości.

    Parametry:
    omega - pulsacja [rad/s]
    k2    - przewodność cieplna warstwy 2 [W/(m·K)]
    alfa2 - dyfuzyjność cieplna warstwy 2 [m^2/s]
    r32   - opór termiczny interfejsu między warstwą 2 a podłożem [m^2·K/W]
    k3    - przewodność cieplna podłoża [W/(m·K)]
    config - konfiguracja (zawiera m.in. d_pump, Q, parametry warstwy 1 i podłoża)
    anisotropy - stosunek K_r/K_z dla warstwy 2 (domyślnie 1.0 – izotropia)

    Zwraca:
    Zespolona amplituda temperatury powierzchniowej [K]
    """
    # Parametry całkowania po λ (radialna liczba falowa)
    lam_max = 35.0 / config.d_pump  # zakres całkowania (do 35/d_pump)
    n_lam = 2000  # liczba punktów całkowania
    lam_vals = np.linspace(0, lam_max, n_lam, endpoint=False)
    delta_lam = lam_vals[1] - lam_vals[0]

    integral = 0.0 + 0.0j

    for lam in lam_vals:
        # Liczby falowe dla każdej warstwy
        sigma1 = thermal_wavevector(lam, omega, config.alfa1, anisotropy=1.0)  # warstwa 1 izotropowa
        sigma2 = thermal_wavevector(lam, omega, alfa2, anisotropy=anisotropy)  # warstwa 2 z anizotropią
        sigma3 = thermal_wavevector(lam, omega, config.alfa3, anisotropy=1.0)  # podłoże izotropowe

        # Macierz całkowita (od góry do podłoża)
        M = np.eye(2, dtype=complex)

        # Warstwa 1
        M1 = layer_transfer_matrix(sigma1, config.k1, config.l1)
        M = M @ M1

        # Interfejs 1-2
        M_int12 = interface_matrix(config.r21)
        M = M @ M_int12

        # Warstwa 2
        M2 = layer_transfer_matrix(sigma2, k2, config.l2)
        M = M @ M2

        # Interfejs 2-3
        M_int23 = interface_matrix(r32)
        M = M @ M_int23

        # Podłoże półnieskończone: impedancja Y = k3 * sigma3
        Y_sub = k3 * sigma3

        # Temperatura na powierzchni dla jednostkowego strumienia q_top = 1 W/m²
        # [T_top; q_top] = M [T_sub; q_sub], a q_sub = Y_sub * T_sub
        # => T_top = (M[0,0] + M[0,1]*Y_sub) / (M[1,0] + M[1,1]*Y_sub)
        theta_lam = (M[0, 0] + M[0, 1] * Y_sub) / (M[1, 0] + M[1, 1] * Y_sub)

        # Funkcja wagowa dla źródła gaussowskiego (transformata Hankela wiązki)
        gauss = np.exp(-(lam * config.d_pump) ** 2 / 4)

        # Przyczynek do całki: theta(λ) * waga(λ) * λ * dλ
        integral += theta_lam * gauss * lam * delta_lam

    # Temperatura powierzchniowa (uwzględnienie mocy całkowitej i konwencji znaku)
    theta_surface = (-config.Q / (2 * np.pi)) * integral
    return theta_surface

