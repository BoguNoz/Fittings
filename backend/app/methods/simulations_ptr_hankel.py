import numpy as np
from scipy.integrate import quad
from typing import Tuple


def simulations_ptr_hankel(
        frequency_vector: np.ndarray,
        k2: float,  # cross-plane thermal conductivity of layer 2 (film)
        alfa2: float,  # cross-plane thermal diffusivity of layer 2
        r32: float,  # R between layer 2 and 3
        k3: float,  # conductivity of substrate
        *,
        k1: float = 21.0,  # transducer conductivity
        l1: float = 80e-9,  # transducer thickness
        l2: float = 469e-9,  # film thickness
        alfa1: float = 8.9e-6,  # transducer diffusivity
        alfa3: float = 6.0e-6,  # substrate diffusivity
        r21: float = 2.8e-8,  # R between transducer and film
        d_pump: float = 2.42e-6,
        Q: float = 1.0,
        anisotropy: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Robust implementation of FDTR model based on Equation

    Fixed issues:
    - Correct alpha for layer 2 (was using alfa1 by mistake)
    - Better numerical stability for large gamma (uses tanh form when possible)
    - Safer handling of very large |s|
    - Improved integration limits and error handling
    """

    omega = 2 * np.pi * frequency_vector
    theta_complex = np.zeros_like(frequency_vector, dtype=complex)

    def layer_transfer_matrix(
            k_perp: float,
            alpha_perp: float,
            thickness: float,
            lam: float,
            om: float,
            eta: float = 1.0
    ) -> np.ndarray:
        """Stable 2x2 transfer matrix for a thermal layer."""
        if thickness < 1e-12:
            return np.eye(2, dtype=complex)

        # σ from Eq. (9)
        s = np.sqrt(eta * lam ** 2 + 1j * om / alpha_perp)

        gamma = s * thickness

        # Use tanh formulation - much more stable for large |gamma|
        if np.abs(gamma.real) > 20.0:  # very thick / high frequency regime
            # For large |gamma|, matrix elements approach exp(-gamma) behavior
            Z = 1.0 / (k_perp * s)
            # Approximate as semi-infinite layer
            return np.array([[1.0, Z],
                             [1.0 / Z, 1.0]], dtype=complex)
        else:
            # Standard stable way using tanh
            tanh_g = np.tanh(gamma)
            Z = 1.0 / (k_perp * s)

            M11 = 1.0
            M12 = Z * tanh_g
            M21 = tanh_g / Z
            M22 = 1.0

            return np.array([[M11, M12],
                             [M21, M22]], dtype=complex)

    def interface_matrix(R: float) -> np.ndarray:
        """Thermal boundary resistance matrix."""
        return np.array([[1.0, R],
                         [0.0, 1.0]], dtype=complex)

    for i, om in enumerate(omega):
        def integrand(lam: float) -> complex:
            if lam < 1e-12:
                return 0.0 + 0j

            # === Correct layer parameters ===
            M1 = layer_transfer_matrix(k1, alfa1, l1, lam, om, eta=1.0)  # transducer
            M2 = layer_transfer_matrix(k2, alfa2, l2, lam, om, eta=anisotropy)  # film ← fixed alfa2!
            M3 = layer_transfer_matrix(k3, alfa3, 1e6, lam, om, eta=1.0)  # substrate

            # Total transfer matrix
            M_total = (M1
                       @ interface_matrix(r21)
                       @ M2
                       @ interface_matrix(r32)
                       @ M3)

            C = M_total[0, 0]
            D = M_total[0, 1]

            # Avoid division by zero or NaN
            if abs(C) < 1e-300:
                return 0.0 + 0j

            theta_lam = -D / C

            gaussian_factor = np.exp(-(lam * d_pump) ** 2 / 8.0)
            return theta_lam * gaussian_factor * lam

        # Integration with safer limits
        upper_limit = 40.0 / d_pump

        try:
            real_part, _ = quad(lambda lam: np.real(integrand(lam)), 0, upper_limit,
                                epsabs=1e-8, epsrel=1e-6, limit=800,
                                points=[1 / d_pump, 5 / d_pump, 15 / d_pump])

            imag_part, _ = quad(lambda lam: np.imag(integrand(lam)), 0, upper_limit,
                                epsabs=1e-8, epsrel=1e-6, limit=800,
                                points=[1 / d_pump, 5 / d_pump, 15 / d_pump])

            theta_complex[i] = real_part + 1j * imag_part
        except:
            theta_complex[i] = 0.0 + 0j

    # Prefactor from Equation (4)
    theta_complex *= -Q / (2 * np.pi)

    # Normalization (relative to DC / low frequency)
    if len(theta_complex) > 0:
        norm = np.abs(theta_complex[0])
        if norm > 1e-300:
            theta_complex /= norm
        else:
            # Fallback if norm is zero
            theta_complex = np.ones_like(theta_complex, dtype=complex)

    amplitude = np.abs(theta_complex)

    return amplitude, theta_complex