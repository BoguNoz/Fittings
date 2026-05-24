# ========== 1. Parametry literaturowe dla 240 nm PEDOT:PSS ==========
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from app.methods.simulations import simulate_single_frequency   # usunięto nieistniejący import
from app.models.ptr_config import PTRConfig

k_cross = 0.18          # cross-plane (W/mK)
k_inplane = 0.35        # in-plane (W/mK)
anisotropy = k_inplane / k_cross   # ≈ 1.94
alfa2 = 1.35e-7         # dyfuzyjność cross-plane (m²/s)
r32 = 1e-8             # opór PEDOT/szkło (m²K/W) – szacunkowy
k3 = 1.0                # przewodność szkła (W/mK)

config = PTRConfig()

# ========== 2. Zakres częstotliwości zgodny z eksperymentem ==========
freq_hz = np.logspace(0, 3.3, 500)   # od 1 Hz do 2000 Hz
omega = 2 * np.pi * freq_hz

print("Obliczanie modelu...")
signal = np.array([simulate_single_frequency(w, k_cross, alfa2, r32, k3, config, anisotropy) for w in omega])
print("Gotowe.")


# Normalizacja (taka sama jak poprzednio: exp(-j45°) * sqrt(f))
offset_deg = 170.0  # Wartość, którą musisz dopasować
norm_factor = np.exp(-1j * np.deg2rad(150.0 + offset_deg)) * np.sqrt(freq_hz)
y_final_norm = signal * norm_factor


amp_model = np.abs(y_final_norm)
phase_model = np.angle(y_final_norm, deg=True)

phase_model_shifted = phase_model + 170.0


# ========== 3. Dane eksperymentalne (z podanej tabeli) ==========
exp_data = np.array([
    [1.00000, 0.02548, -4.35637],
    [1.28835, 0.02616, -3.39999],
    [1.65985, 0.02674, -2.45112],
    [2.13847, 0.02731, -1.41552],
    [2.75510, 0.02808, -0.36862],
    [3.54954, 0.02913,  0.67978],
    [4.57305, 0.03018,  1.72143],
    [5.89170, 0.03173,  2.77691],
    [7.59057, 0.03308,  3.77641],
    [9.77933, 0.03550,  4.58947],
    [12.59921, 0.03744,  5.36728],
    [16.23221, 0.04029,  5.91922],
    [20.91279, 0.04277,  6.26038],
    [26.94302, 0.04487,  6.50078],
    [34.71208, 0.04907,  6.53104],
    [44.72136, 0.05253,  6.21927],
    [57.61683, 0.05631,  5.66040],
    [74.23072, 0.06088,  4.78837],
    [95.63525, 0.06533,  3.24142],
    [123.21181, 0.07002,  1.27054],
    [158.74011, 0.07513, -1.16878],
    [204.51304, 0.08084, -4.40046],
    [263.48466, 0.08735, -8.45728],
    [339.46083, 0.09238, -12.65437],
    [437.34483, 0.09773, -17.54789],
    [563.45382, 0.10105, -22.58780],
    [725.92652, 0.10387, -27.21693],
    [935.24845, 0.10452, -31.14528],
    [1204.92865, 0.09521, -25.98794],
    [1552.37151, 0.11971, -32.07072],
    [2000.00000, 0.11756, 335.77399]
])
exp_freq = exp_data[:, 0]
exp_amp = exp_data[:, 1]
exp_phase = exp_data[:, 2]
# Poprawka fazy: wartości >180° zamieniamy na ujemne (np. 335.77° → -24.23°)
exp_phase[exp_phase > 180] -= 360

# Normalizacja amplitudy do wartości przy najniższej częstotliwości (dla porównania kształtu)
amp_model_norm = amp_model / amp_model[0]
exp_amp_norm = exp_amp / exp_amp[0]

# ========== 4. Wykresy ==========
plt.figure(figsize=(10, 8))

# Amplituda
plt.subplot(2, 1, 1)
plt.loglog(freq_hz, amp_model_norm, 'b-', linewidth=2, label='Model (norm. exp(-j45°)*√f)')
plt.loglog(exp_freq, exp_amp_norm, 'ro', markersize=4, label='Eksperyment')
plt.ylabel('Amplituda znormalizowana (j.u.)')
plt.grid(True, which='both', alpha=0.5)
plt.legend()
plt.title('Porównanie modelu z danymi eksperymentalnymi – PEDOT:PSS 240 nm')

# Faza
plt.subplot(2, 1, 2)
plt.semilogx(freq_hz, phase_model, 'b-', linewidth=2, label='Model')
plt.semilogx(exp_freq, exp_phase, 'ro', markersize=4, label='Eksperyment')
plt.xlabel('Częstotliwość (Hz)')
plt.ylabel('Faza (°)')
plt.grid(True, which='both', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()

# Opcjonalnie: wyświetlenie kilku pierwszych wartości dla kontroli
print("\nModel amplituda:", amp_model)
print("Eksperyment amplituda", exp_amp_norm)
print("\nModel faza:", phase_model)
print("Eksperyment faza", exp_phase)
