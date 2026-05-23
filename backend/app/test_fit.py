# ========== 2. Parametry literaturowe dla 240 nm PEDOT:PSS ==========
import numpy as np
from matplotlib import pyplot as plt

from app.methods.simulations import simulate_single_frequency, simulate_single_frequency_2
from app.models.ptr_config import PTRConfig

k_cross = 0.18          # cross-plane (W/mK)
k_inplane = 0.35        # in-plane (W/mK)
anisotropy = k_inplane / k_cross   # ≈ 1.94
alfa2 = 1.35e-7         # dyfuzyjność cross-plane (m²/s)
r32 = 1e-8              # opór PEDOT/szkło (m²K/W) – szacunkowy
k3 = 1.0                # przewodność szkła (W/mK) – nie jest dopasowywany

config = PTRConfig()

# Zakres częstotliwości od 1 kHz do 1 MHz
freq_hz = np.logspace(3, 6, 50)
omega = 2 * np.pi * freq_hz

print("Obliczanie modelu...")
signal = np.array([simulate_single_frequency(w, k_cross, alfa2, r32, k3, config, anisotropy) for w in omega])
print("Gotowe.")

norm_factor = np.exp(-1j * np.deg2rad(45.0)) * np.sqrt(freq_hz)
y_final_norm = signal * norm_factor

print("Raw abs:", np.abs(signal))
print("Raw phase:", np.angle(signal))

print("Normalized abs:", np.abs(y_final_norm))
print("Normalized phase:", np.angle(y_final_norm))

# ========== 3. Wykresy surowego sygnału ==========
plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.loglog(freq_hz, np.abs(y_final_norm), 'b.-', linewidth=2)
plt.ylabel('Amplituda |θ_TR| (j.u.)')
plt.grid(True, which='both', alpha=0.5)
plt.title('Surowy sygnał modelu Henkela 3D (PEDOT:PSS 240 nm, parametry literaturowe)')

plt.subplot(2, 1, 2)
plt.semilogx(freq_hz, np.angle(y_final_norm, deg=True), 'r.-', linewidth=2)
plt.xlabel('Częstotliwość (Hz)')
plt.ylabel('Faza (°)')
plt.grid(True, which='both', alpha=0.5)

plt.tight_layout()
plt.show()