import numpy as np
from scipy.integrate import quad

from app.methods.simulations_ptr import simulations_ptr


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
        d_pump: float = 2.42e-6,
        Q: float = 1.0,
        anisotropy: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    omega = 2 * np.pi * frequency_vector
    y_complex = np.zeros_like(frequency_vector, dtype=complex)

    def integrand(lam: float, om: float) -> complex:
        if lam < 1e-12:
            return 0.0 + 0j

        s1 = np.sqrt(anisotropy * lam ** 2 + 1j * om / alfa1)
        s2 = np.sqrt(anisotropy * lam ** 2 + 1j * om / alfa2)
        s3 = np.sqrt(anisotropy * lam ** 2 + 1j * om / alfa3)

        t1 = np.tan(s1 * l1)
        t2 = np.tan(s2 * l2)

        ro12 = (k1 * s1) / (k2 * s2)
        ro21 = (k2 * s2) / (k1 * s1)

        A = (1 + r21 * k1 * s1 * t1 + ro12 * t1 * t2 +
             r32 * (k1 * s1 * t1 + k2 * s2 * t2 - r21 * k1 * s1 * k2 * s2 * t1 * t2))

        B = (-t1 / (k1 * s1) - t2 / (k2 * s2) - r21 -
             r32 * (1 + ro21 * t1 * t2 - r21 * k2 * s2 * t2))

        G = -k1 * s1 * t1 - k2 * s2 * t2 - r21 * k1 * s1 * k2 * s2 * t1 * t2
        D = 1 + ro21 * t1 * t2 + r21 * k2 * s2 * t2

        theta = -(G - k3 * s3 * A) / (D - k3 * s3 * B)

        gaussian = np.exp(-(lam * d_pump) ** 2 / 8.0)
        return theta * gaussian * lam

    for i, om in enumerate(omega):
        upper_limit = 40.0 / d_pump  # zwiększone
        real_part, _ = quad(lambda lam: np.real(integrand(lam, om)), 0, upper_limit,
                            epsabs=1e-8, epsrel=1e-5, limit=400, points=[5 / d_pump])
        imag_part, _ = quad(lambda lam: np.imag(integrand(lam, om)), 0, upper_limit,
                            epsabs=1e-8, epsrel=1e-5, limit=400, points=[5 / d_pump])

        y_complex[i] = real_part + 1j * imag_part

    y_complex *= -Q / (2 * np.pi)
    y_complex /= np.sqrt(omega + 1e-30)

    _, y1d_complex = simulations_ptr(frequency_vector, k2, alfa2, r32, k3,
                                     k1=k1, l1=l1, l2=l2, alfa1=alfa1,
                                     alfa3=alfa3, r21=r21)
    scale = np.abs(y1d_complex[0]) / (np.abs(y_complex[0]) + 1e-300)
    y_complex *= scale

    return np.abs(y_complex), y_complex