import numpy as np
import matplotlib.pyplot as plt

from app.processors.ptr_processors_builder import FittingProcessorBuilder
from app.models.ptr_config import PTRConfig


# ============================== 1. CONFIGURATION ==============================
config = PTRConfig(
)

# ============================== 2. DATA LOADING & FITTING ==============================
print("Starting PTR data processing...\n")

result = (FittingProcessorBuilder()
            .load_dat_file("data/PEDO-25.dat", sample_name="X32B")
            .load_config(config)
            .set_starting_point_count(15)      # więcej startów = lepsze szanse
            .build()
            .process())

# ============================== 3. PRINT FIT RESULTS ==============================
print("=== PTR FITTING SUMMARY ===")
print(f"k2 (cross-plane)     : {result.k2:.4f} W/(m·K)")
print(f"k_parallel           : {result.k_parallel:.4f} W/(m·K)")
print(f"alfa2                : {result.alfa2:.2e} m²/s")
print(f"r32 (boundary)       : {result.r32:.2e} m²·K/W")
print(f"Anisotropy           : {result.anisotropy:.2f}")
print(f"Residual norm        : {result.res_norm:.6f}")
print(f"Best resnorm         : {result.res_norm:.6f}")

# ============================== 4. PLOTTING ==============================
def plot_ptr_result(result, filename=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Użyj result.frequency_hz zamiast frequency_vector
    ax1.loglog(result.frequency_hz, result.exp_amp, 'o', markersize=4, label='Experiment')
    ax1.loglog(result.frequency_hz, result.model_amp, '-', lw=2, label='Model')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, which='both', ls='--')

    ax2.semilogx(result.frequency_hz, result.exp_phase_deg, 'o', markersize=4, label='Experiment')
    ax2.semilogx(result.frequency_hz, result.model_phase_deg, '-', lw=2, label='Model')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (deg)')
    ax2.legend()
    ax2.grid(True, which='both', ls='--')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


plot_ptr_result(result)

print("results:", result)

# Po fitowaniu dodaj:
print("Frequency range:", result.frequency_hz.min(), "–", result.frequency_hz.max())
print("Phase at highest freq - Exp:", result.exp_phase_deg[-5:].mean())
print("Phase at highest freq - Model:", result.model_phase_deg[-5:].mean())

print("Successfully completed fitting and plotting.")