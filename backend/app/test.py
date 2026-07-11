import multiprocessing as mp
import warnings

import matplotlib.pyplot as plt
from scipy.integrate import IntegrationWarning

from app.processors.ptr_processors_builder import FittingProcessorBuilder
from app.models.ptr_config import PTRConfig

import numpy as np

def save_fit_to_dat(result, filename="fit_result.dat"):
    """
    Zapisuje wyniki fitowania do pliku DAT.

    Kolumny:
    Frequency[Hz]  ModelAmplitude  ModelPhase[deg]
    """
    data = np.column_stack([
        result.frequency_hz,
        result.model_amp,
        result.model_phase_deg
    ])

    header = (
        "Frequency_Hz\tModelAmplitude\tModelPhase_deg\n"
        "# Generated from PTR fitting"
    )

    np.savetxt(
        filename,
        data,
        fmt="%.10e",
        delimiter="\t",
        header=header,
        comments=""
    )

    print(f"Fit saved to: {filename}")

warnings.filterwarnings("ignore", category=IntegrationWarning)
# ============================== KONFIGURACJA ==============================
config = PTRConfig(
    l1=50e-9,
    k1=21.9,                # Ti
    alfa1=9.4e-6,           # Ti
    l2=300e-9,
    alfa3=1.2e-5,           # szafir
    k3=35.0,                # szafir
    r21=6.1e-8,             # R_th z tabeli dla 0% Mg
    d_pump=3.0e-3,          # poprawiona średnica plamki
    Q=1.0,
    anisotropy=4.0,         # przykładowo, jeśli H = k_perp/k_par, to anizotropia = 1/H
)

# ============================== GŁÓWNY KOD ==============================
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    warnings.filterwarnings("ignore", category=IntegrationWarning)
    print("Starting PTR data processing...\n")

    result = (FittingProcessorBuilder()
                .load_dat_file("data/PTR_Z150620Akwarc.dat", sample_name="X32B")
                .load_config(config)
                .set_starting_point_count(10)
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

    save_fit_to_dat(result)

    # ============================== DIAGNOSTYKA ==============================
    print("=== DIAGNOSTYKA KOŃCOWA ===")
    print("Frequency range:", result.frequency_hz.min(), "–", result.frequency_hz.max())
    print(f"Phase at highest freq - Exp: {result.exp_phase_deg[-5:].mean():.2f}°")
    print(f"Phase at highest freq - Model: {result.model_phase_deg[-5:].mean():.2f}°")
    print("Successfully completed fitting and plotting.")

