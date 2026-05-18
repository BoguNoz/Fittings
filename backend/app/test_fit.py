import numpy as np
from app.methods.correct_ptr_data import correct_ptr_data
from app.methods.fit_ptr_multi_start import fit_ptr_multi_start
from app.models.ptr_config import PTRConfig

# ---------- raw experimental data (frequency, amplitude, phase) ----------
raw_data = np.array([
    [1.00000,   0.02548,  -4.35637],
    [1.28835,   0.02616,  -3.39999],
    [1.65985,   0.02674,  -2.45112],
    [2.13847,   0.02731,  -1.41552],
    [2.75510,   0.02808,  -0.36862],
    [3.54954,   0.02913,   0.67978],
    [4.57305,   0.03018,   1.72143],
    [5.89170,   0.03173,   2.77691],
    [7.59057,   0.03308,   3.77641],
    [9.77933,   0.03550,   4.58947],
    [12.59921,  0.03744,   5.36728],
    [16.23221,  0.04029,   5.91922],
    [20.91279,  0.04277,   6.26038],
    [26.94302,  0.04487,   6.50078],
    [34.71208,  0.04907,   6.53104],
    [44.72136,  0.05253,   6.21927],
    [57.61683,  0.05631,   5.66040],
    [74.23072,  0.06088,   4.78837],
    [95.63525,  0.06533,   3.24142],
    [123.21181, 0.07002,   1.27054],
    [158.74011, 0.07513,  -1.16878],
    [204.51304, 0.08084,  -4.40046],
    [263.48466, 0.08735,  -8.45728],
    [339.46083, 0.09238, -12.65437],
    [437.34483, 0.09773, -17.54789],
    [563.45382, 0.10105, -22.58780],
    [725.92652, 0.10387, -27.21693],
    [935.24845, 0.10452, -31.14528],
    [1204.92865,0.09521, -25.98794],
    [1552.37151,0.11971, -32.07072],
    [2000.00000,0.11756, 335.77399]
])

freq_raw = raw_data[:, 0]
amp_raw = raw_data[:, 1]
phase_raw = raw_data[:, 2]

# ---------- clean data ----------
freq, amp, phase_unwrapped = correct_ptr_data(freq_raw, amp_raw, phase_raw,
                                              remove_amp_outliers=True,
                                              outlier_threshold=0.05)
print("Frequencies after cleaning:", len(freq))

# ---------- configure model ----------
config = PTRConfig(
    l2=240e-9,        # 240 nm PEDOT:PSS
    rhoc=2.5e6,       # from your requirement
    d_pump=2.42e-6,   # 1/e² radius from PDF
)

# ---------- run fitting ----------
phys_params = {
    'l2': config.l2,
    'k1': config.k1,
    'l1': config.l1,
    'alfa1': config.alfa1,
    'alfa3': config.alfa3,
    'r21': config.r21,
    'd_pump': config.d_pump,
    'Q': config.Q,
    'rhoc': config.rhoc,
    'weight_exponent': config.weight_exponent,
    'phase_weight': config.phase_weight,
}

result = fit_ptr_multi_start(
    frequency_vector=freq,
    exp_amp=amp,
    exp_phase=phase_unwrapped,   # already unwrapped!
    n_starts=35,
    **phys_params
)

# ---------- output ----------
print("\n=== FITTING RESULTS (Hankel model) ===")
print(f"k2 (cross-plane)     : {result.k2:.4f} W/m·K")
print(f"α2 (cross-plane)     : {result.alfa2:.2e} m²/s")
print(f"k∥ (in-plane)        : {result.k_parallel:.4f} W/m·K")
print(f"anisotropy (k∥/k⊥)   : {result.anisotropy:.2f}")
print(f"r32 (boundary resist.): {result.r32:.2e} m²·K/W")
print(f"k3 (substrate)       : {result.k3:.4f} W/m·K")
print(f"φ0 (phase offset)    : {result.phi0_deg:.3f}°")
print(f"Residual norm        : {result.res_norm:.6f}")