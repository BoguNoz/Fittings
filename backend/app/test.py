import numpy as np
import matplotlib.pyplot as plt

from app.processors.ptr_processors_builder import FittingProcessorBuilder
from app.models.ptr_config import PTRConfig

# ============================== KROK 1: KONFIGURACJA MODELU ==============================
# Inicjalizacja podstawowej konfiguracji.
# (Wewnętrznie anizotropia i parametry podłoża są zamrożone zgodnie z wcześniejszymi ustaleniami)
config = PTRConfig()

# ============================== KROK 2: ŁADOWANIE DANYCH I FITOWANIE ==============================
print("Starting PTR data processing...\n")

# Uruchomienie buildera: wczytuje plik, aplikuje filtry i przeprowadza wielostartową optymalizację
result = (FittingProcessorBuilder()
            .load_dat_file("data/PEDO-1.dat", sample_name="X32B")
            .load_config(config)
            .set_starting_point_count(20) # Ilość punktów startowych algorytmu
            .build()
            .process())

# ============================== KROK 3: RAPORT DLA PROMOTORA ==============================
# Generowanie uporządkowanego podsumowania zawierającego sztywne założenia,
# wyliczone parametry cross-plane, in-plane oraz statystyki błędu (R2 i residua).
print("\n" + "="*45)
print("           === RAPORT DLA PROMOTORA ===")
print("="*45)
print(f"Zadana anizotropia (sztywna)            : {result.anisotropy:.2f}")
print(f"Przewodnictwo cross-plane (k_cross)     : {result.k2:.4f} W/(m·K)")
print(f"Dyfuzyjność cross-plane (alfa)          : {result.alfa2:.2e} m²/s")
print(f"Rezystancja termiczna interfejsu (R_th) : {result.r32:.2e} m²·K/W")
print("-" * 45)
print(f"Wyprowadzone przewodnictwo in-plane     : {result.k_parallel:.4f} W/(m·K)")
print("-" * 45)
print(f"R² (Amplituda)                          : {result.r2_amp:.4f}")
print(f"R² (Faza)                               : {result.r2_phase:.4f}")
print(f"Suma kwadratów reszt (Residua)          : {result.res_norm:.6e}")
print("="*45 + "\n")

# ============================== KROK 4: RYSOWANIE WYKRESÓW ==============================
# Wizualizacja dopasowania krzywych wygenerowanego modelu do danych eksperymentalnych
def plot_ptr_result(result, filename=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Wykres Amplitudy (skala log-log)
    ax1.loglog(result.frequency_hz, result.exp_amp, 'ro', markersize=4, label='Experiment')
    ax1.loglog(result.frequency_hz, result.model_amp, 'b-', lw=2, label='Model')
    ax1.set_ylabel('Amplitude (a.u.)')
    ax1.legend()
    ax1.grid(True, which='both', ls='--', alpha=0.5)

    # Wykres Fazy (skala semi-log)
    ax2.semilogx(result.frequency_hz, result.exp_phase_deg, 'ro', markersize=4, label='Experiment')
    ax2.semilogx(result.frequency_hz, result.model_phase_deg, 'b-', lw=2, label='Model')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (deg)')
    ax2.legend()
    ax2.grid(True, which='both', ls='--', alpha=0.5)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

plot_ptr_result(result)

# ============================== KROK 5: KONTROLA STABILNOŚCI FAZY NA KOŃCACH ==============================
# Weryfikacja jak dobrze model domyka się na najwyższych częstotliwościach
print("=== DIAGNOSTYKA KOŃCOWA ===")
print("Frequency range:", result.frequency_hz.min(), "–", result.frequency_hz.max())
print(f"Phase at highest freq - Exp: {result.exp_phase_deg[-5:].mean():.2f}°")
print(f"Phase at highest freq - Model: {result.model_phase_deg[-5:].mean():.2f}°")
print("Successfully completed fitting and plotting.")