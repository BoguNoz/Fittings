import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ====================== DANE ======================
data = np.loadtxt('data/32B.dat')  # lub wklej ręcznie
freq_khz = data[:, 0]
amp_exp = data[:, 1]
phase_exp = data[:, 2]

# Usuń ostatnie punkty (zaszumione)
mask = (freq_khz <= 800) & (phase_exp < 100)  # obcinamy mocno
freq = freq_khz[mask] * 1000  # do Hz
amp = amp_exp[mask]
phase = phase_exp[mask]

print(f"Używam {len(freq)} punktów do {freq.max() / 1000:.0f} kHz")


# ====================== PROSTY MODEL 1D (lepszy na start) ======================
def ptr_model_1d(omega, k2, alfa2, R32, k3=1.0, L2=240e-9):
    """Bardzo uproszczony model 1D transfer matrix"""
    sigma2 = np.sqrt(1j * omega / alfa2)
    sigma3 = np.sqrt(1j * omega / (k3 / 2.3e6))  # przybliżenie szkła

    # Prosty model dla PEDOT na szkle z R32
    Z2 = 1 / (k2 * sigma2)
    Z3 = 1 / (k3 * sigma3)

    # Transmission line approximation
    tanh_term = np.tanh(sigma2 * L2)
    theta = (Z3 + R32) / (Z2 * (1 + (Z3 + R32) / Z2 * tanh_term)) * tanh_term  # uproszczone

    signal = theta / np.sqrt(omega + 1e-30)
    return signal


def residual(p, freq, amp, phase):
    k2, log_alfa, log_R32, log_aniso = p
    alfa2 = 10 ** log_alfa
    R32 = 10 ** log_R32
    aniso = 10 ** log_aniso
    k2 = max(0.05, k2)

    omega = 2 * np.pi * freq

    y_complex = ptr_model_1d(omega, k2, alfa2, R32)

    # Normalizacja + phase offset
    y_norm = y_complex / y_complex[0]
    phi0 = p[4] if len(p) > 4 else 0.0
    y_norm *= np.exp(1j * np.deg2rad(phi0))

    exp_complex = amp / amp[0] * np.exp(1j * np.unwrap(np.deg2rad(phase)))

    diff = y_norm - exp_complex
    w = (freq / freq.max()) ** 0.6
    residuals = np.concatenate([np.real(diff) * w, np.imag(diff) * w * 12])
    return residuals


# ====================== FITTING ======================
p0 = [0.20, np.log10(8e-8), np.log10(2e-7), np.log10(2.5), 40.0]
bounds = ([0.05, -10, -9, 0, -100], [1.0, -6, -5, 4, 100])

res = least_squares(residual, p0, bounds=bounds, args=(freq, amp, phase),
                    ftol=1e-10, xtol=1e-10, max_nfev=2000)

print("\n=== WYNIKI ===")
print(f"k2 (cross)     = {res.x[0]:.4f} W/mK")
print(f"alfa2          = {10 ** res.x[1]:.2e} m²/s")
print(f"R32            = {10 ** res.x[2]:.2e} m²K/W")
print(f"Anisotropy     = {10 ** res.x[3]:.2f}")
print(f"phi0           = {res.x[4]:.2f} deg")
print(f"Cost           = {res.cost:.4f}")

# Plot
y_best = ptr_model_1d(2 * np.pi * freq, res.x[0], 10 ** res.x[1], 10 ** res.x[2])
y_best /= y_best[0]
y_best *= np.exp(1j * np.deg2rad(res.x[4]))

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.loglog(freq, amp / amp[0], 'bo', label='Exp')
plt.loglog(freq, np.abs(y_best), 'r-', label='Model')
plt.legend();
plt.grid()

plt.subplot(2, 1, 2)
plt.semilogx(freq, phase, 'bo', label='Exp')
plt.semilogx(freq, np.angle(y_best, deg=True), 'r-', label='Model')
plt.legend();
plt.grid();
plt.xlabel('Frequency (Hz)');
plt.ylabel('Phase (deg)')
plt.tight_layout()
plt.show()