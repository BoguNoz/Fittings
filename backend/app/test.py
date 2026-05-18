import numpy as np
import matplotlib.pyplot as plt

from app.fitting_processors.ptr_processors_builder import FittingProcessorBuilder
from app.methods.simulations_ptr_hankel import simulations_ptr_hankel
from app.models.ptr_config import PTRConfig


# ============================== 1. CONFIGURATION ==============================
config = PTRConfig(
    l2=240e-9,
    alfa3=0.5e-7,
    rhoc=2.5e6,
    d_pump=2.42e-6,
)

# ============================== 2. DATA LOADING & FITTING ==============================
print("Starting PTR data processing...\n")

result = (FittingProcessorBuilder()
            .load_dat_file("data/PEDO-1.dat", sample_name="X32B")
            .load_config(config)
            .set_starting_point_count(25)
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
fig, axs = plt.subplots(3, 1, figsize=(11, 14))

freq = result.frequency_vector
freq_log = np.log10(freq)

other_color = '#666666'
alpha_val = 0.28
linewidth_other = 0.85

print(f"Total attempts: {len(result.all_results)}")
sorted_attempts = sorted(result.all_results, key=lambda x: x['resnorm'])
other_to_plot = sorted_attempts[1:15] + sorted_attempts[-7:]
print(f"Attempting to plot {len(other_to_plot)} alternative models")

# Wspólne argumenty stałe dla modelu
model_kwargs = dict(
    k1=config.k1, l1=config.l1, l2=config.l2,
    alfa1=config.alfa1, alfa3=config.alfa3, r21=config.r21,
    d_pump=config.d_pump, Q=config.Q
)

# ==================== 1. RAW AMPLITUDE ====================
axs[0].plot(freq, result.exp_amp, 'ko', markersize=4, label='Experiment', zorder=15)

plotted = 0
for attempt in other_to_plot:
    try:
        _, y_complex = simulations_ptr_hankel(
            freq,
            k2=attempt['k2'],
            alfa2=attempt['alfa2'],
            r32=attempt['r32'],
            k3=attempt['k3'],
            anisotropy=attempt['anisotropy'],
            **model_kwargs
        )
        if np.any(np.isnan(y_complex)) or np.any(np.isinf(y_complex)):
            continue

        y_norm = (y_complex / y_complex[0]) * np.exp(1j * np.deg2rad(attempt['phi0_deg']))
        gain = np.mean(result.exp_amp[:5] / np.abs(y_norm[:5]) + 1e-12)
        model_amp = np.abs(y_norm) * gain

        axs[0].plot(freq, model_amp, color=other_color, alpha=alpha_val,
                    linewidth=linewidth_other, zorder=1)
        plotted += 1
    except Exception:
        continue

axs[0].plot(freq, result.model_amp, 'r-', linewidth=3.2, label='Best Model', zorder=20)
axs[0].set_xscale('log')
axs[0].set_ylabel('Amplitude [a.u.]')
axs[0].set_title(f'PTR Amplitude (Best + {plotted} other attempts)')
axs[0].grid(True, alpha=0.5)
axs[0].legend()

# ==================== 2. NORMALIZED AMPLITUDE ====================
model_norm = result.model_amp / result.model_amp[0]
exp_norm = result.exp_amp / result.exp_amp[0]

axs[1].plot(freq_log, np.log10(exp_norm), 'ko', markersize=4, label='Experiment', zorder=15)

for attempt in other_to_plot:
    try:
        _, y_complex = simulations_ptr_hankel(
            freq,
            k2=attempt['k2'], alfa2=attempt['alfa2'],
            r32=attempt['r32'], k3=attempt['k3'],
            anisotropy=attempt['anisotropy'],
            **model_kwargs
        )
        if np.any(np.isnan(y_complex)) or np.any(np.isinf(y_complex)):
            continue
        y_norm = (y_complex / y_complex[0]) * np.exp(1j * np.deg2rad(attempt['phi0_deg']))
        m_norm = np.abs(y_norm) / np.abs(y_norm[0])
        axs[1].plot(freq_log, np.log10(m_norm), color=other_color, alpha=alpha_val,
                    linewidth=linewidth_other, zorder=1)
    except Exception:
        continue

axs[1].plot(freq_log, np.log10(model_norm), 'r-', linewidth=3.2, label='Best Model', zorder=20)
axs[1].set_xlabel('log10(Frequency) [Hz]')
axs[1].set_ylabel('log10(Normalized Amplitude)')
axs[1].set_title('Normalized Amplitude (log-log)')
axs[1].grid(True, alpha=0.5)
axs[1].legend()

# ==================== 3. PHASE ====================
axs[2].plot(freq_log, result.exp_phase_deg, 'ko', markersize=4, label='Experiment', zorder=15)

for attempt in other_to_plot:
    try:
        _, y_complex = simulations_ptr_hankel(
            freq,
            k2=attempt['k2'], alfa2=attempt['alfa2'],
            r32=attempt['r32'], k3=attempt['k3'],
            anisotropy=attempt['anisotropy'],
            **model_kwargs
        )
        if np.any(np.isnan(y_complex)) or np.any(np.isinf(y_complex)):
            continue
        y_norm = (y_complex / y_complex[0]) * np.exp(1j * np.deg2rad(attempt['phi0_deg']))
        phase_other = np.unwrap(np.angle(y_norm)) * 180 / np.pi
        axs[2].plot(freq_log, phase_other, color=other_color, alpha=alpha_val,
                    linewidth=linewidth_other, zorder=1)
    except Exception:
        continue

axs[2].plot(freq_log, result.model_phase_deg, 'r-', linewidth=3.2, label='Best Model', zorder=20)
axs[2].set_xlabel('log10(Frequency) [Hz]')
axs[2].set_ylabel('Phase [deg]')
axs[2].set_title('PTR Phase')
axs[2].grid(True, alpha=0.5)
axs[2].legend()

plt.tight_layout()
plt.show()

print(f"Successfully plotted {plotted} alternative models.")