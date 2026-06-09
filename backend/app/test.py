import multiprocessing as mp
import warnings

import matplotlib.pyplot as plt
from scipy.integrate import IntegrationWarning

from app.processors.ptr_processors_builder import FittingProcessorBuilder
from app.models.ptr_config import PTRConfig
warnings.filterwarnings("ignore", category=IntegrationWarning)
# ============================== KONFIGURACJA ==============================
config = PTRConfig(
    anisotropy=3.67,
    l2=240e-9 * 0.9,
    d_pump=1.8e-6
)

# ============================== GŁÓWNY KOD ==============================
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    warnings.filterwarnings("ignore", category=IntegrationWarning)
    print("Starting PTR data processing...\n")

    result = (FittingProcessorBuilder()
                .load_dat_file("data/PEDO-25.dat", sample_name="X32B")
                .load_config(config)
                .set_starting_point_count(80)
                .build()
                .process())

    # ============================== RAPORT ==============================
    print("\n" + "="*45)
    print("           === RAPORT ===")
    print("="*45)
    print(f"Zadana anizotropia (sztywna)            : {result.anisotropy:.2f}")
    print(f"Przewodnictwo cross-plane (k_cross)     : {result.k2:.10f} W/(m·K)")
    print(f"Dyfuzyjność cross-plane (alfa)          : {result.alfa2:.2e} m²/s")
    print(f"Rezystancja termiczna interfejsu (R_th) : {result.r32:.2e} m²·K/W")
    print("-" * 45)
    print(f"Wyprowadzone przewodnictwo in-plane     : {result.k_parallel:.4f} W/(m·K)")
    print("-" * 45)
    print(f"R² (Amplituda)                          : {result.r2_amp:.4f}")
    print(f"R² (Faza)                               : {result.r2_phase:.4f}")
    print(f"Suma kwadratów reszt (Residua)          : {result.res_norm:.6e}")
    print("="*45 + "\n")

    # ============================== WYKRESY ==============================
    def plot_ptr_result(result, filename=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.loglog(result.frequency_hz, result.exp_amp, 'ro', markersize=4, label='Experiment')
        ax1.loglog(result.frequency_hz, result.model_amp, 'b-', lw=2, label='Model')
        ax1.set_ylabel('Amplitude (a.u.)')
        ax1.legend()
        ax1.grid(True, which='both', ls='--', alpha=0.5)

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

    # ============================== DIAGNOSTYKA ==============================
    print("=== DIAGNOSTYKA KOŃCOWA ===")
    print("Frequency range:", result.frequency_hz.min(), "–", result.frequency_hz.max())
    print(f"Phase at highest freq - Exp: {result.exp_phase_deg[-5:].mean():.2f}°")
    print(f"Phase at highest freq - Model: {result.model_phase_deg[-5:].mean():.2f}°")
    print("Successfully completed fitting and plotting.")