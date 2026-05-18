import numpy as np
from scipy.integrate import quad
from typing import Tuple


def simulations_ptr_hankel(
        frequency_vector: np.ndarray,
        k2: float,          # cross-plane thermal conductivity of the film (W/m·K)
        alfa2: float,       # cross-plane thermal diffusivity of the film (m²/s)
        r32: float,         # film/substrate thermal boundary resistance (m²·K/W)
        k3: float,          # substrate thermal conductivity (W/m·K)
        *,
        k1: float = 200.0,           # transducer thermal conductivity (Al)
        l1: float = 50e-9,           # transducer thickness (m)
        l2: float = 240e-9,          # film thickness (m)
        alfa1: float = 8.2e-5,       # transducer diffusivity (m²/s)
        alfa3: float = 0.5e-6,       # substrate diffusivity (glass)
        r21: float = 1.0e-8,         # transducer/film interface resistance (m²·K/W)
        d_pump: float = 2.42e-6,     # pump 1/e² radius (m)
        Q: float = 1.0,              # normalized pump power
        anisotropy: float = 1.0,     # k_parallel / k_perp for the film
        **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full PTR response using Hankel transform (radial heat flow).
    Equation (4) / (9) from the paper:
    θ = -Q/(2π) ∫ [D/C * exp(-(λ·d_pump)²/8) * λ dλ] / √(ω)

    Returns raw complex signal (NOT normalized).
    """
    omega = 2 * np.pi * frequency_vector
    theta_complex = np.zeros_like(frequency_vector, dtype=complex)

    # ----- layer transfer matrix with tanh for stability -----
    def layer_matrix(kperp, alpha_perp, thickness, lam, om, eta=1.0):
        if thickness < 1e-12:
            return np.eye(2, dtype=complex)
        s = np.sqrt(eta * lam**2 + 1j * om / alpha_perp)
        gamma = s * thickness
        # semi-infinite approximation
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

    # ----- Hankel integral for each frequency -----
    for idx, om in enumerate(omega):
        # upper limit: 20/d_pump is enough because exp(-(λ·d)²/8) decays rapidly
        upper = 20.0 / d_pump
        # break points for adaptive integration
        points = [0.5/d_pump, 2.0/d_pump, 8.0/d_pump, 15.0/d_pump]

        def integrand(lam):
            if lam < 1e-30:
                return 0.0 + 0j
            M1 = layer_matrix(k1, alfa1, l1, lam, om, eta=1.0)
            M2 = layer_matrix(k2, alfa2, l2, lam, om, eta=anisotropy)
            M3 = layer_matrix(k3, alfa3, 1e6, lam, om, eta=1.0)  # semi-infinite substrate
            M = M1 @ interface_matrix(r21) @ M2 @ interface_matrix(r32) @ M3
            C, D = M[0, 0], M[0, 1]
            if abs(C) < 1e-300:
                return 0.0 + 0j
            theta_lam = -D / C
            gauss = np.exp(-(lam * d_pump)**2 / 8.0)
            return theta_lam * gauss * lam

        try:
            real_part, _ = quad(lambda lam: np.real(integrand(lam)),
                                0, upper, limit=200, points=points,
                                epsabs=1e-9, epsrel=1e-6)
            imag_part, _ = quad(lambda lam: np.imag(integrand(lam)),
                                0, upper, limit=200, points=points,
                                epsabs=1e-9, epsrel=1e-6)
            theta_complex[idx] = real_part + 1j * imag_part
        except Exception:
            theta_complex[idx] = 0.0 + 0j

    # Prefactor and 1/√ω factor (essential for correct frequency behaviour)
    theta_complex *= -Q / (2 * np.pi)
    theta_complex /= np.sqrt(omega + 1e-30)
    #print(f"f={frequency_vector[0]:.1f} Hz, theta={theta_complex[0]:.3e}")

    amplitude = np.abs(theta_complex)
    return amplitude, theta_complex   # no internal normalization