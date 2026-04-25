import numpy as np
from scipy.optimize import least_squares

from app.methods.ptr_residual import ptr_residual
from app.methods.simulations_ptr import simulations_ptr
from app.methods.simulations_ptr_hankel import simulations_ptr_hankel
from app.models.ptr_fit_result import PTRFitResult # Upewnij się, że ten import działa


def fit_ptr(
        frequency_vector: np.ndarray,
        exp_amp: np.ndarray,
        exp_phase: np.ndarray,
        phase_units: str = "auto",
        use_hankel: bool = False,
        **phys_params
) -> PTRFitResult:
    """
    Main fitting routine for PTR parameters (k2, alfa2, r32, k3).
    Fits the complex response by aligning both model and experiment
    to their respective first frequency points.
    """

    # --- Initial Guess and Bounds (log10 scale for physical params) ---
    p0 = np.array([np.log10(10.0), np.log10(3e-6), np.log10(1e-8), np.log10(3.0), 0.0])

    # Extended bounds to account for high-diffusivity ZnO and prevent hitting walls
    lb = np.array([np.log10(1e-3), np.log10(1e-9), np.log10(1e-10), np.log10(0.1), -360.0])
    ub = np.array([np.log10(500.0), np.log10(1e-3), np.log10(1e-4), np.log10(100.0), 360.0])

    # --- Unit Detection ---
    used_units = phase_units.lower()
    if used_units == "auto":
        res_deg = least_squares(ptr_residual, p0, bounds=(lb, ub),
                                args=(frequency_vector, exp_amp, exp_phase, "deg", use_hankel),
                                kwargs=phys_params)
        res_rad = least_squares(ptr_residual, p0, bounds=(lb, ub),
                                args=(frequency_vector, exp_amp, exp_phase, "rad", use_hankel),
                                kwargs=phys_params)

        if res_deg.cost <= res_rad.cost:
            res, used_units = res_deg, "deg"
        else:
            res, used_units = res_rad, "rad"
    else:
        res = least_squares(ptr_residual, p0, bounds=(lb, ub),
                            args=(frequency_vector, exp_amp, exp_phase, used_units, use_hankel),
                            kwargs=phys_params)

    # --- Post-processing and Result Extraction ---
    pfit = res.x
    k2, alfa2, r32, k3 = 10 ** pfit[0], 10 ** pfit[1], 10 ** pfit[2], 10 ** pfit[3]
    phi0_deg = pfit[4]

    # Generate final model response for visualization
    if use_hankel:
        _, y_complex = simulations_ptr_hankel(frequency_vector, k2, alfa2, r32, k3, **phys_params)
    else:
        _, y_complex = simulations_ptr(frequency_vector, k2, alfa2, r32, k3, **phys_params)

    # Model: Normalized to (1+0j) at f[0], then rotated by phi0
    y_model_final = (y_complex / y_complex[0]) * np.exp(1j * np.deg2rad(phi0_deg))
    model_amp = np.abs(y_model_final)
    model_phase_deg = np.unwrap(np.angle(y_model_final)) * 180 / np.pi

    # Experiment: Reconstruct complex signal and normalize identically to the model
    exp_phase_rad = np.deg2rad(exp_phase) if used_units == "deg" else exp_phase
    exp_complex_raw = exp_amp * np.exp(1j * exp_phase_rad)
    exp_final = (exp_complex_raw / exp_complex_raw[0]) * np.exp(1j * np.deg2rad(phi0_deg))

    exp_phase_deg_plot = np.unwrap(np.angle(exp_final)) * 180 / np.pi

    # --- RETURN OBJECT INSTEAD OF DICTIONARY ---
    return PTRFitResult(
        k2=k2,
        alfa2=alfa2,
        r32=r32,
        k3=k3,
        phi0_deg=phi0_deg,
        res_norm=float(2 * res.cost),
        model_amp=model_amp,
        model_phase_deg=model_phase_deg,
        exp_phase_deg=exp_phase_deg_plot,
        phase_units=used_units,
        pfit=pfit,
        exit_flag=int(res.status),
        frequency_vector=frequency_vector,
        l2=phys_params.get('l2', 469e-9),
    )