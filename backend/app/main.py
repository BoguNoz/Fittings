import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from app.methods.fit_ptr import fit_ptr
from app.models.ptr_fit_result import PTRFitResult


# ============================== 1. WCZYTANIE DANYCH ==============================
def load_ptr_data(filename: str):
    """Wczytuje dane PTR z pliku .dat"""
    data = np.loadtxt(filename)

    # Zakładamy strukturę: kolumna 0 = freq [kHz], kolumna 1 = amplitude, kolumna 2 = phase
    frequency_kHz = data[:, 0]
    exp_amp = data[:, 1]
    exp_phase = data[:, 2]

    frequency_vector = frequency_kHz * 1000.0  # zamiana kHz → Hz

    print(f"Wczytano dane: {len(frequency_vector)} punktów")
    print(f"Zakres częstotliwości: {frequency_vector.min():.1f} – {frequency_vector.max():.1f} Hz")

    return frequency_vector, exp_amp, exp_phase


# ============================== 2. GŁÓWNY PROGRAM ==============================
# ------------------- Wczytanie danych -------------------
frequency_vector, exp_amp, exp_phase = load_ptr_data("twoje_dane.dat")
# Możesz zmienić nazwę pliku np. "30A.dat", "tr1.dat" itd.

# ------------------- Dopasowanie modelu -------------------
print("\nRozpoczynam dopasowanie modelu PTR...")

result: PTRFitResult = fit_ptr(
    frequency_vector=frequency_vector,
    exp_amp=exp_amp,
    exp_phase=exp_phase,
    phase_units="auto"  # automatycznie sprawdzi deg/rad
)

# ------------------- Wyświetlenie wyników -------------------
result.print_summary()

# Szczegółowe wartości
print(f"\nDokładniejsze wyniki:")
print(f"k2   = {result.k2:.6f} W/(m·K)")
print(f"alfa2 = {result.alfa2:.6e} m²/s")
print(f"r32  = {result.r32:.6e} m²·K/W")
print(f"resnorm = {result.resnorm:.6f}")

# ------------------- Wykresy -------------------
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

freq_log = np.log10(frequency_vector)

# Amplitude plot
axs[0].plot(freq_log, np.log10(exp_amp / exp_amp[0]), 'ko', label='Experiment')
axs[0].plot(freq_log, np.log10(result.model_amp), 'r-', linewidth=2, label='Model')
axs[0].set_xlabel('log10(Frequency) [Hz]')
axs[0].set_ylabel('log10(Normalized Amplitude)')
axs[0].grid(True, alpha=0.5)
axs[0].legend()

# Phase plot
axs[1].plot(freq_log, result.exp_phase_deg, 'ko', label='Experiment')
axs[1].plot(freq_log, result.model_phase_deg, 'r-', linewidth=2, label='Model')
axs[1].set_xlabel('log10(Frequency) [Hz]')
axs[1].set_ylabel('Phase [deg]')
axs[1].grid(True, alpha=0.5)
axs[1].legend()

plt.tight_layout()
plt.show()

# ------------------- Opcjonalnie: zapis wyników -------------------
print("\nZapisywanie wyników do pliku...")
np.savetxt('ptr_fit_results.dat',
           np.column_stack((frequency_vector,
                            result.model_amp,
                            result.model_phase_deg,
                            exp_amp,
                            result.exp_phase_deg)),
           header='Frequency_Hz   Model_Amp   Model_Phase_deg   Exp_Amp   Exp_Phase_deg',
           comments='#')

# Możesz też zapisać cały obiekt (np. pickle)
import pickle

with open('ptr_fit_result.pkl', 'wb') as f:
    pickle.dump(result, f)

print("Gotowe! Wyniki zapisane.")