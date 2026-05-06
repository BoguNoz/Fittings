import numpy as np
from scipy.integrate import quad
from typing import Tuple


def simulations_ptr_hankel(
        frequency_vector: np.ndarray,
        k2: float,          # cross-plane thermal conductivity of the film (layer 2)
        alfa2: float,       # cross-plane thermal diffusivity of the film (layer 2)
        r32: float,         # thermal boundary resistance between film and substrate (m²K/W)
        k3: float,          # thermal conductivity of the substrate (layer 3)
        *,
        k1: float = 21.0,           # thermal conductivity of the transducer (layer 1)
        l1: float = 80e-9,          # thickness of the transducer (m)
        l2: float = 469e-9,         # thickness of the film (m)
        alfa1: float = 8.9e-6,      # thermal diffusivity of the transducer (m²/s)
        alfa2_conf = -1,
        alfa3: float = 6.0e-6,      # thermal diffusivity of the substrate (m²/s)
        r21: float = 2.8e-8,        # thermal boundary resistance between transducer and film (m
        d_pump: float = 2.42e-6,    # pump beam 1/e² radius (m)
        Q: float = 1.0,             # normalized pump power
        anisotropy: float = 1.0,    # anisotropy ratio k_parallel / k_perp for the film
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the complex PTR (Photothermal Radiometry) response using the full Hankel transform
    (radial heat flow) based on Equation (4) from the paper "Photothermal Infrared Radiometry
    and Thermoreflectance—Unique Strategy for Thermal Transport Characterization of Nanolayers".

    This implementation is specifically tuned for PTR measurements.

    Key features:
    - Stable 2x2 thermal transfer matrix formulation using tanh for numerical robustness
    - Correct frequency-dependent factor 1/sqrt(ω) required for proper PTR phase and amplitude behavior
    - Gaussian beam profile in Hankel space: exp(-(λ * d_pump)²/8)
    - Support for thermal anisotropy in the film layer only

    Returns
    -------
    amplitude : np.ndarray
        Normalized amplitude of the PTR signal
    y_complex : np.ndarray
        Complex PTR response (used for phase calculation)
    """

    omega = 2 * np.pi * frequency_vector
    theta_complex = np.zeros_like(frequency_vector, dtype=complex)
    alfa2 = alfa2_conf if alfa2_conf != -1 else alfa2

    def layer_transfer_matrix(
            k_perp: float,
            alpha_perp: float,
            thickness: float,
            lam: float,
            om: float,
            eta: float = 1.0
    ) -> np.ndarray:
        """
        Stable 2x2 thermal transfer matrix for a single layer using tanh formulation.
        This avoids numerical overflow that occurs with direct sinh/cosh at high frequencies.
        """
        if thickness < 1e-12:
            return np.eye(2, dtype=complex)

        # σ from Eq. (9) in the paper
        s = np.sqrt(eta * lam**2 + 1j * om / alpha_perp)
        gamma = s * thickness

        # Semi-infinite layer approximation for very large |gamma|
        if np.abs(gamma.real) > 25.0 or np.abs(gamma.imag) > 25.0:
            Z = 1.0 / (k_perp * s)
            return np.array([[1.0, Z],
                             [1.0 / Z, 1.0]], dtype=complex)

        # Standard stable tanh form
        tanh_g = np.tanh(gamma)
        Z = 1.0 / (k_perp * s)

        return np.array([[1.0,          Z * tanh_g],
                         [tanh_g / Z,   1.0]], dtype=complex)

    def interface_matrix(R: float) -> np.ndarray:
        """Thermal boundary resistance (Kapitza resistance) interface matrix."""
        return np.array([[1.0, R],
                         [0.0, 1.0]], dtype=complex)

    for i, om in enumerate(omega):
        def integrand(lam: float) -> complex:
            if lam < 1e-12:
                return 0.0 + 0j

            # Build total transfer matrix: M1 @ R21 @ M2 @ R32 @ M3
            M1 = layer_transfer_matrix(k1, alfa1, l1, lam, om, eta=1.0)                    # transducer
            M2 = layer_transfer_matrix(k2, alfa2, l2, lam, om, eta=anisotropy)             # film (with anisotropy)
            M3 = layer_transfer_matrix(k3, alfa3, 1e6, lam, om, eta=1.0)                   # thick substrate

            M_total = (M1 @ interface_matrix(r21) @
                       M2 @ interface_matrix(r32) @ M3)

            C = M_total[0, 0]
            D = M_total[0, 1]

            if abs(C) < 1e-300:
                return 0.0 + 0j

            theta_lam = -D / C                                 # θ = -D/C  (as in Eq. 4 and 10)
            gaussian_factor = np.exp(-(lam * d_pump)**2 / 8.0) # Gaussian beam in Hankel domain

            return theta_lam * gaussian_factor * lam

        # Integration limits and points optimized for Gaussian beam
        upper_limit = 80.0 / d_pump

        try:
            real_part, _ = quad(lambda lam: np.real(integrand(lam)), 0, upper_limit,
                                epsabs=1e-9, epsrel=1e-6, limit=1500,
                                points=[0.5/d_pump, 2.0/d_pump, 8.0/d_pump, 25.0/d_pump])

            imag_part, _ = quad(lambda lam: np.imag(integrand(lam)), 0, upper_limit,
                                epsabs=1e-9, epsrel=1e-6, limit=1500,
                                points=[0.5/d_pump, 2.0/d_pump, 8.0/d_pump, 25.0/d_pump])

            theta_complex[i] = real_part + 1j * imag_part

        except Exception:
            theta_complex[i] = 0.0 + 0j

    # Prefactor from Equation (4) in the paper
    theta_complex *= -Q / (2 * np.pi)

    # The 1/sqrt(ω) factor is essential for correct frequency dependence
    # of both amplitude and phase in photothermal radiometry (matches 1D version)
    theta_complex /= np.sqrt(omega + 1e-30)

    # Normalize amplitude relative to the first (lowest) frequency point
    # This is consistent with how experimental data is usually processed
    if len(theta_complex) > 0:
        norm = np.abs(theta_complex[0])
        if norm > 1e-300:
            theta_complex /= norm
        else:
            theta_complex = np.ones_like(theta_complex, dtype=complex)

    amplitude = np.abs(theta_complex)

    return amplitude, theta_complex