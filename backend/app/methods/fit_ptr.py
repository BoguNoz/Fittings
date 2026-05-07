import numpy as np
from scipy.optimize import least_squares
from app.methods.ptr_residual import ptr_residual
from app.methods.simulations_ptr import simulations_ptr
from app.models.ptr_fit_result import PTRFitResult

def fit_ptr(
        frequency_vector: np.ndarray,
        exp_amp: np.ndarray,
        exp_phase: np.ndarray,
        **phys_params
) -> PTRFitResult:
    """
    Core routine to fit PTR thermal parameters by minimizing complex residuals.

    Args:
        frequency_vector: Modulation frequencies [Hz].
        exp_amp: Measured PTR amplitude.
        exp_phase: Measured PTR phase in degrees.
        **phys_params: Additional physical constants passed to the simulation.
    """
    # --- Optimization Setup ---
    # Log10 scale used for k2, alfa2, r32, k3 to maintain positivity and handle magnitudes
    p0 = np.array([np.log10(1.0), np.log10(1e-7), np.log10(1e-8), np.log10(3.0), 0.0])
    lb = np.array([np.log10(1e-3), np.log10(1e-10), np.log10(1e-10), np.log10(0.1), -360.0])
    ub = np.array([np.log10(500.0), np.log10(1e-3), np.log10(1e-4), np.log10(100.0), 360.0])

    # --- Run Least Squares Optimization ---
    res = least_squares(
        ptr_residual, p0, bounds=(lb, ub),
        args=(frequency_vector, exp_amp, exp_phase),
        kwargs=phys_params,
        ftol=1e-8, xtol=1e-8
    )

    # --- Process Fitted Parameters ---
    pfit = res.x
    k2, alfa2, r32, k3 = 10 ** pfit[0], 10 ** pfit[1], 10 ** pfit[2], 10 ** pfit[3]
    phi0_deg = pfit[4]

    # --- Generate Scaled Model for Results ---
    _, y_complex = simulations_ptr(frequency_vector, k2, alfa2, r32, k3, **phys_params)

    # Normalize and rotate model by fitted phase offset
    y_model_norm = (y_complex / y_complex[0]) * np.exp(1j * np.deg2rad(phi0_deg))
    model_phase_deg = np.unwrap(np.angle(y_model_norm)) * 180 / np.pi

    # Align model amplitude with experimental magnitude (gain matching)
    gain = np.mean(exp_amp[:5] / np.abs(y_model_norm[:5]))
    model_amp_scaled = np.abs(y_model_norm) * gain

    # --- Prepare Experimental Data for Plotting ---
    # Construct complex representation to perform robust unwrapping
    exp_complex = exp_amp * np.exp(1j * np.deg2rad(exp_phase))
    exp_norm = (exp_complex / exp_complex[0]) * np.exp(1j * np.deg2rad(phi0_deg))
    exp_phase_deg_plot = np.unwrap(np.angle(exp_norm)) * 180 / np.pi

    return PTRFitResult(
        k2=k2,
        alfa2=alfa2,
        r32=r32,
        k3=k3,
        phi0_deg=phi0_deg,
        res_norm=float(2 * res.cost),
        model_amp=model_amp_scaled,
        exp_amp=exp_amp,
        model_phase_deg=model_phase_deg,
        exp_phase_deg=exp_phase_deg_plot,
        pfit=pfit,
        exit_flag=int(res.status),
        frequency_vector=frequency_vector,
        l2=phys_params.get('l2', 469e-9),
    )