import numpy as np
import matplotlib.pyplot as plt

from app.fitting_processors.ptr_processors_builder import FittingProcessorBuilder
from app.models.ptr_config import PTRConfig


# ============================== 1. CONFIGURATION ==============================
# Set the main ZnO layer thickness here (obtained from RBS analysis)
# Change this value according to your sample (in nanometers)
config = PTRConfig(
    l2=469.5e-9,        # Thickness of main ZnO layer in meters
    # Other parameters can be adjusted if needed:
    # k1=21.0, l1=80e-9, alfa1=8.9e-6, alfa3=6.0e-6, r21=2.8e-8
)


# ============================== 2. DATA LOADING & FITTING ==============================
print("Starting PTR data processing...\n")

# Create and run the fitting using the builder
result = (FittingProcessorBuilder()
          .load_dat_file("data/32B.dat", sample_name="X32B")   # ← CHANGE filename and sample name
          .load_config(config)
          .apply_phase_correction(45)
          .build().process())

# ============================== 3. PRINT FIT RESULTS ==============================
print("=== PTR FITTING SUMMARY ===")
print(f"Sample:               {result.sample_name}")
print(f"k2 (thermal conductivity): {result.k2:.4f} W/(m·K)")
print(f"alfa2 (thermal diffusivity): {result.alfa2:.2e} m²/s")
print(f"r32 (thermal boundary resistance): {result.r32:.2e} m²·K/W")
print(f"Phase offset (phi0):   {result.phi0_deg:.3f} deg")
print(f"Residual norm:         {result.res_norm:.6f}")
print(f"Phase units used:      {result.phase_units}\n")


# ============================== 4. PLOTTING - THREE SUBPLOTS ==============================
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

freq_log = np.log10(result.frequency_vector)

# ------------------- 1. Amplitude (linear scale) -------------------
axs[0].plot(result.frequency_vector, result.model_amp, 'r-', linewidth=2, label='Model')
axs[0].plot(result.frequency_vector, result.model_amp[0] * (result.model_amp / result.model_amp[0]),
            'ko', markersize=4, label='Experiment')   # normalized for comparison
axs[0].set_xscale('log')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('PTR Amplitude')
axs[0].grid(True, alpha=0.5)
axs[0].legend()

# ------------------- 2. Normalized Amplitude (log-log) -------------------
axs[1].plot(freq_log, np.log10(result.model_amp), 'r-', linewidth=2, label='Model')
axs[1].plot(freq_log, np.log10(result.model_amp / result.model_amp[0]),
            'ko', markersize=4, label='Experiment')
axs[1].set_xlabel('log10(Frequency) [Hz]')
axs[1].set_ylabel('log10(Normalized Amplitude)')
axs[1].set_title('Normalized Amplitude (log-log)')
axs[1].grid(True, alpha=0.5)
axs[1].legend()

# ------------------- 3. Phase -------------------
axs[2].plot(freq_log, result.model_phase_deg, 'r-', linewidth=2, label='Model')
axs[2].plot(freq_log, result.exp_phase_deg, 'ko', markersize=4, label='Experiment')
axs[2].set_xlabel('log10(Frequency) [Hz]')
axs[2].set_ylabel('Phase [deg]')
axs[2].set_title('PTR Phase')
axs[2].grid(True, alpha=0.5)
axs[2].legend()

plt.tight_layout()
plt.show()


# ============================== 5. FINAL MESSAGE ==============================
print("Fitting completed successfully. Plots displayed.")