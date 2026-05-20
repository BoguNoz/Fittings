import numpy as np
import matplotlib.pyplot as plt

from app.fitting_processors.ptr_processors_builder import FittingProcessorBuilder
from app.methods.simulations_ptr_hankel import simulations_ptr_hankel
from app.models.ptr_config import PTRConfig


# ============================== 1. CONFIGURATION ==============================
config = PTRConfig(
    l2=240e-9,
    k1=200.0,
    l1=50e-9,
    alfa1=8.2e-5,
    alfa3=0.5e-7,
    r21=1.0e-8,
    rhoc=2.0e6,
    d_pump=2.42e-6,
    Q=1.0,
    weight_exponent=0.5,
    phase_weight=1.5
)

# ============================== 2. DATA LOADING & FITTING ==============================
print("Starting PTR data processing...\n")

result = (FittingProcessorBuilder()
            .load_dat_file("data/PEDO-25.dat", sample_name="X32B")
            .load_config(config)
            .set_starting_point_count(1)
            .build().process())

print("Exp amp first 5:", result.exp_amp[:5])
print("Exp amp last 5:", result.exp_amp[-5:])
print("Exp amp ratio max/min:", result.exp_amp.max() / result.exp_amp.min())
print("Frequency range:", result.frequency_vector.min(), "-", result.frequency_vector.max())

# ============================== 3. PRINT FIT RESULTS ==============================
print("=== PTR FITTING SUMMARY ===")
print(f"k2 (thermal conductivity): {result.k2:.4f} W/(m·K)")
print(f"k_parallel: {result.k_parallel:.4f} W/(m·K)")
print(f"alfa2 (thermal diffusivity): {result.alfa2:.2e} m²/s")
print(f"r32 (thermal boundary resistance): {result.r32:.2e} m²·K/W")
print(f"Phase offset (phi0): {result.phi0_deg:.3f} deg")
print(f"Residual norm: {result.res_norm:.6f}")

# ============================== 4. PLOTTING ==============================
import matplotlib.pyplot as plt

def plot_ptr_result(result, filename=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.loglog(result.frequency_vector, result.exp_amp, 'o', label='exp')
    ax1.loglog(result.frequency_vector, result.model_amp, '-', label='model')
    ax1.set_ylabel('Amplitude ratio')
    ax1.legend()
    ax1.grid(True, which='both', ls='--')

    ax2.semilogx(result.frequency_vector, result.exp_phase_deg, 'o', label='exp')
    ax2.semilogx(result.frequency_vector, result.model_phase_deg, '-', label='model')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (deg)')
    ax2.legend()
    ax2.grid(True, which='both', ls='--')

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

plot_ptr_result(result)

print(f"Successfully plotted.")