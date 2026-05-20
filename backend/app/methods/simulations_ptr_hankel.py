import numpy as np
from scipy.integrate import quad
from typing import Tuple


def simulations_ptr_hankel(
        frequency_vector: np.ndarray,
        k2: float,
        alfa2: float,
        r32: float,
        k3: float,
        *,
        k1: float = 200.0,
        l1: float = 50e-9,
        l2: float = 240e-9,
        alfa1: float = 8.2e-5,
        alfa3: float = 0.5e-6,
        r21: float = 1.0e-8,
        d_pump: float = 2.42e-6,
        Q: float = 1.0,
        anisotropy: float = 1.0,
        **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full PTR response using Hankel transform.
    Returns raw complex signal (NOT normalized).
    """
    omega = 2 * np.pi * frequency_vector
    theta_complex = np.zeros_like(frequency_vector, dtype=complex)

    def layer_matrix(kperp, alpha_perp, thickness, lam, om, eta=1.0):
        if thickness < 1e-12:
            return np.eye(2, dtype=complex)
        s_sq = eta * lam**2 + 1j * om / alpha_perp
        # unikamy ujemnych części rzeczywistych pod pierwiastkiem
        s = np.sqrt(s_sq) if np.real(s_sq) >= 0 else np.sqrt(-s_sq) * 1j
        gamma = s * thickness
        if np.abs(gamma.real) > 25 or np.abs(gamma.imag) > 25:
            Z = 1.0 / (kperp * s)
            return np.array([[1.0, Z],
                             [1.0 / Z, 1.0]], dtype=complex)
        tanh_g = np.tanh(gamma)
        Z = 1.0 / (kperp * s)
        return np.array([[1.0,          Z * tanh_g],
                         [tanh_g / Z,   1.0]], dtype=complex)

    def interface_matrix(R):
        return np.array([[1.0, R],
                         [0.0, 1.0]], dtype=complex)

    for idx, om in enumerate(omega):
        upper = 50.0 / d_pump  # było 20.0
        log_points = np.logspace(np.log10(0.5 / d_pump), np.log10(upper), 15)
        points = np.unique(np.concatenate(([0.5 / d_pump, 1.0 / d_pump, 2.0 / d_pump, 5.0 / d_pump],
                                           log_points)))

        def integrand(lam):
            if lam < 1e-30:
                return 0.0 + 0j
            try:
                M1 = layer_matrix(k1, alfa1, l1, lam, om, eta=1.0)
                M2 = layer_matrix(k2, alfa2, l2, lam, om, eta=anisotropy)
                M3 = layer_matrix(k3, alfa3, 1e6, lam, om, eta=1.0)
                M = M1 @ interface_matrix(r21) @ M2 @ interface_matrix(r32) @ M3
                C, D = M[0, 0], M[0, 1]
                if abs(C) < 1e-300:
                    return 0.0 + 0j
                theta_lam = -D / C
                gauss = np.exp(-(lam * d_pump) ** 2 / 8.0)
                return theta_lam * gauss * lam
            except Exception:
                return 0.0 + 0j

        try:
            real_part, _ = quad(lambda lam: np.real(integrand(lam)),
                                0, upper, limit=500, points=points,
                                epsabs=1e-12, epsrel=1e-8)
            imag_part, _ = quad(lambda lam: np.imag(integrand(lam)),
                                0, upper, limit=500, points=points,
                                epsabs=1e-12, epsrel=1e-8)
            theta_complex[idx] = real_part + 1j * imag_part
        except Exception:
            theta_complex[idx] = 0.0 + 0j

    theta_complex *= -Q / (2 * np.pi)
    theta_complex /= np.sqrt(omega + 1e-30)

    theta_complex = np.nan_to_num(theta_complex, nan=0.0, posinf=0.0, neginf=0.0)

    amplitude = np.abs(theta_complex)
    return amplitude, theta_complex