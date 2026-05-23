# ========== Model FDTR dla PEDOT:PSS 480 nm – parametry z artykułu ==========
# Beam spot = 2.42 µm (zgodnie z artykułem)
# Plik: plot_pedot_pss_article.py

import numpy as np
import matplotlib.pyplot as plt

from app.methods.simulations import simulate_single_frequency
from app.models.ptr_config import PTRConfig

# ====================== PARAMETRY Z ARTYKUŁU (Tabela 7 + Tabela 5) ======================
k_cross     = 0.182          # cross-plane thermal conductivity [W/m·K]
k_inplane   = 0.67           # in-plane thermal conductivity [W/m·K]
anisotropy  = k_inplane / k_cross   # ≈ 3.68
alfa2       = 1.48254e-7     # cross-plane thermal diffusivity [m²/s]
r32         = 1.6245e-9      # thermal boundary resistance PEDOT:PSS / glass [m²K/W]
k3          = 1.0            # glass substrate

# Konfiguracja warstwy (z Tabeli 5)
config = PTRConfig(
    l1=20e-9,           # Au (20 nm)
    k1=105.0,           # Au
    alfa1=1.23e-5,
    l2=480e-9,          # PEDOT:PSS – 480 nm
    rhoc2=k_cross / alfa2,   # obliczone z k i alfa
    alfa3=0.34e-6,      # glass
    k3=k3,
    r21=1e-9,           # Au / PEDOT (przybliżenie)
    d_pump=2.40e-6,     # beam spot
    Q=1.0,
    weight_exponent=0.8,
    phase_weight=1.2
)

print(f"Anisotropy η = {anisotropy:.2f}")
print(f"ρC₂ = {config.rhoc2:.3e} J/m³K")

# ====================== ZAKRES CZĘSTOTLIWOŚCI ======================
freq_hz = np.logspace(3, 6.5, 80)   # 1 kHz – ~3 MHz
omega = 2 * np.pi * freq_hz

# ====================== OBLICZANIE MODELU ======================
print("Obliczanie modelu Hankela 3D...")
signal = np.array([
    simulate_single_frequency(w, k_cross, alfa2, r32, k3, config, anisotropy)
    for w in omega
])

# ====================== NORMALIZACJA (jak w Twoim fitterze) ======================
norm_factor = np.exp(-1j * np.deg2rad(45.0)) / np.sqrt(freq_hz)
y_norm = signal * norm_factor

# ====================== WYKRESY ======================
plt.figure(figsize=(12, 9))

# --- Amplituda ---
plt.subplot(2, 1, 1)
plt.loglog(freq_hz, np.abs(signal), 'b.-', label='Surowy sygnał (model)', linewidth=2.2)
plt.loglog(freq_hz, np.abs(y_norm), 'r--', label='Znormalizowany (amp / √f)', linewidth=2)
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('|θ_TR| (j.u.)')
plt.title('FDTR – PEDOT:PSS 480 nm (beam spot = 2.42 µm)\n'
          f'k⊥ = {k_cross} W/m·K, k∥ = {k_inplane} W/m·K, η = {anisotropy:.2f}')
plt.grid(True, which='both', alpha=0.6)
plt.legend()

# --- Faza ---
plt.subplot(2, 1, 2)
plt.semilogx(freq_hz, np.angle(signal, deg=True), 'b.-', label='Surowa faza', linewidth=2.2)
plt.semilogx(freq_hz, np.angle(y_norm, deg=True), 'r--', label='Znormalizowana faza (-45°)', linewidth=2)
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Faza [°]')
plt.grid(True, which='both', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()

# Opcjonalnie: zapis do pliku
# plt.savefig('FDTR_PEDOT_PSS_480nm_beam_2.42um.png', dpi=300, bbox_inches='tight')