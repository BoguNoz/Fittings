import numpy as np


def simulate_tr_single_layer(t: np.ndarray, k: float, alfa: float, config) -> np.ndarray:
    """
    Symulacja odpowiedzi przejściowej (TR) dla jednej warstwy (półprzestrzeni) 3D.
    t: wektor czasu w sekundach
    k: przewodność cieplna
    alfa: dyfuzyjność cieplna
    """
    # Pojemność cieplna objętościowa rho*c = k / alfa
    rhoc = k / alfa
    w0 = config.d_pump  # Promień plamki lasera

    # Zapobieganie dzieleniu przez zero dla t=0
    t = np.maximum(t, 1e-12)

    # Klasyczne rozwiązanie 3D dla powierzchni półprzestrzeni z plamką gaussowską
    # T(t) prop_to 1 / (rho * c * sqrt(pi * alfa * t) * (1 + 8 * alfa * t / w0^2))
    # Skalowanie zależy od całkowitej energii impulsu Q

    denominator = rhoc * np.sqrt(np.pi * alfa * t) * (1.0 + (8.0 * alfa * t) / (w0 ** 2))

    # Stała amplituda podziału energii
    T_surf = config.Q / denominator
    return T_surf