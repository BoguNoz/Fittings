import numpy as np
from scipy.integrate import quad


def simulations_ptr_hankel(
        frequency_vector: np.ndarray,
        k2: float,
        alfa2: float,
        r32: float,
        k3: float,
        *,
        k1: float = 21.0,
        l1: float = 80e-9,
        l2: float = 469e-9,
        alfa1: float = 8.9e-6,
        alfa3: float = 6.0e-6,
        r21: float = 2.8e-8,
        d_pump: float = 2.42e-6,  # Pump beam diameter (1/e²)
        Q: float = 1.0,  # Pump power (normalized)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Corrected implementation of Equation (4) using the Hankel transform.
    Calculates the photothermal response in a multi-layer system.

    Returns:
        tuple: (amplitude, complex_signal)
    """
    # Angular frequency conversion
    omega = 2 * np.pi * frequency_vector
    y_complex = np.zeros_like(frequency_vector, dtype=complex)

    def integrand(lam: float, om: float) -> complex:
        """
        Kernel of the Hankel transform integration.
        lam: spatial frequency variable
        om: angular frequency
        """
        if lam < 1e-12:
            return 0.0 + 0j

        # Thermal wave numbers for layers 1, 2, and 3
        s1 = np.sqrt(1j * om / alfa1)
        s2 = np.sqrt(1j * om / alfa2)
        s3 = np.sqrt(1j * om / alfa3)

        # Tangent factors for finite thickness layers
        t1 = np.tan(s1 * l1)
        t2 = np.tan(s2 * l2)

        # Thermal effusivity/impedance ratios
        ro12 = (k1 * s1) / (k2 * s2)
        ro21 = (k2 * s2) / (k1 * s1)

        # Transfer matrix coefficients (based on the 1D multi-layer model)
        A = (1 + r21 * k1 * s1 * t1 + ro12 * t1 * t2 +
             r32 * (k1 * s1 * t1 + k2 * s2 * t2 - r21 * k1 * s1 * k2 * s2 * t1 * t2))

        B = (-t1 / (k1 * s1) - t2 / (k2 * s2) - r21 -
             r32 * (1 + ro21 * t1 * t2 - r21 * k2 * s2 * t2))

        G = -k1 * s1 * t1 - k2 * s2 * t2 - r21 * k1 * s1 * k2 * s2 * t1 * t2
        D = 1 + ro21 * t1 * t2 + r21 * k2 * s2 * t2

        # Surface temperature coefficient in frequency domain
        theta = -(G - k3 * s3 * A) / (D - k3 * s3 * B)

        # Gaussian beam profile factor from Equation (4)
        gaussian = np.exp(- (lam * d_pump) ** 2 / 8.0)

        return theta * gaussian * lam

    for i, om in enumerate(omega):
        # Integrate real and imaginary parts separately for numerical stability
        real_part, _ = quad(lambda lam: np.real(integrand(lam, om)), 0, 1e8,
                            epsabs=1e-10, epsrel=1e-8, limit=400)
        imag_part, _ = quad(lambda lam: np.imag(integrand(lam, om)), 0, 1e8,
                            epsabs=1e-10, epsrel=1e-8, limit=400)

        y_complex[i] = real_part + 1j * imag_part

    # Apply scaling factors from Equation (4)
    y_complex *= -Q / (2 * np.pi)

    # Diffusion factor for correct phase behavior in the frequency domain
    y_complex /= np.sqrt(1j * omega + 1e-30)

    amplitude = np.abs(y_complex)
    return amplitude, y_complex