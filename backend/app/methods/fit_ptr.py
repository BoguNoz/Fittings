import numpy as np
from scipy.optimize import least_squares

from app.methods.ptr_residual import ptr_residual
from app.methods.simulations_ptr import simulations_ptr
from app.methods.simulations_ptr_hankel import simulations_ptr_hankel
from app.models.ptr_fit_result import PTRFitResult


def fit_ptr(
        frequency_vector: np.ndarray,
        exp_amp: np.ndarray,
        exp_phase: np.ndarray,
        phase_units: str = "auto",
        use_hankel: bool = False,
        **phys_params
) -> PTRFitResult:
    """
    Main fitting routine for PTR (Photothermal Radiometry) data.

    Fits thermal parameters (k2, alfa2, r32) by minimizing the complex residual
    between the model and experimental data (amplitude + phase).
    """
    # --- Initial Guess and Bounds (parameters in log10 scale for better optimization) ---
    p0 = np.array([np.log10(10.0),  # k2
                   np.log10(3e-6),  # alfa2
                   np.log10(1e-8),  # r32
                   np.log10(3.0),  # k3
                   0.0])  # phi0 (phase offset in degrees)

    lb = np.array([np.log10(1e-3), np.log10(1e-9), np.log10(1e-10), np.log10(0.1), -360.0])
    ub = np.array([np.log10(500.0), np.log10(1e-3), np.log10(1e-4), np.log10(100.0), 360.0])

    # --- Automatic phase unit detection (degrees vs radians) ---
    used_units = phase_units.lower()
    if used_units == "auto":
        res_deg = least_squares(ptr_residual, p0, bounds=(lb, ub),
                                args=(frequency_vector, exp_amp, exp_phase, "deg", use_hankel),
                                kwargs=phys_params)

        res_rad = least_squares(ptr_residual, p0, bounds=(lb, ub),
                                args=(frequency_vector, exp_amp, exp_phase, "rad", use_hankel),
                                kwargs=phys_params)

        # Choose the better fit
        if res_deg.cost <= res_rad.cost:
            res, used_units = res_deg, "deg"
        else:
            res, used_units = res_rad, "rad"
    else:
        res = least_squares(ptr_residual, p0, bounds=(lb, ub),
                            args=(frequency_vector, exp_amp, exp_phase, used_units, use_hankel),
                            kwargs=phys_params)

    # --- Extract fitted parameters ---
    pfit = res.x
    k2 = 10 ** pfit[0]
    alfa2 = 10 ** pfit[1]
    r32 = 10 ** pfit[2]
    k3 = 10 ** pfit[3]
    phi0_deg = pfit[4]

    # --- Generate complex model response ---
    if use_hankel:
        _, y_complex = simulations_ptr_hankel(
            frequency_vector, k2, alfa2, r32, k3, **phys_params
        )
    else:
        _, y_complex = simulations_ptr(
            frequency_vector, k2, alfa2, r32, k3, **phys_params
        )

    # Model: normalize to first point + apply phase offset
    y_model_norm = (y_complex / y_complex[0]) * np.exp(1j * np.deg2rad(phi0_deg))
    model_amp_norm = np.abs(y_model_norm)
    model_phase_deg = np.unwrap(np.angle(y_model_norm)) * 180 / np.pi

    # --- Scale model amplitude to match experimental scale ---
    # This is crucial for the raw amplitude plot to look correct
    exp_phase_rad = np.deg2rad(exp_phase) if used_units == "deg" else exp_phase
    exp_complex_raw = exp_amp * np.exp(1j * exp_phase_rad)

    # Calculate gain using first N points (robust against noise at high frequencies)
    N = min(8, len(exp_amp))
    gain = np.mean(exp_amp[:N] / model_amp_norm[:N])

    model_amp_scaled = model_amp_norm * gain

    # --- Prepare experimental phase for plotting (with same phase offset) ---
    exp_final = (exp_complex_raw / exp_complex_raw[0]) * np.exp(1j * np.deg2rad(phi0_deg))
    exp_phase_deg_plot = np.unwrap(np.angle(exp_final)) * 180 / np.pi

    # --- Return result object ---
    return PTRFitResult(
        k2=k2,
        alfa2=alfa2,
        r32=r32,
        k3=k3,
        phi0_deg=phi0_deg,
        res_norm=float(2 * res.cost),
        model_amp=model_amp_scaled,  # scaled to experimental amplitude
        exp_amp=exp_amp,  # raw experimental amplitude
        model_phase_deg=model_phase_deg,
        exp_phase_deg=exp_phase_deg_plot,
        phase_units=used_units,
        pfit=pfit,
        exit_flag=int(res.status),
        frequency_vector=frequency_vector,
        l2=phys_params.get('l2', 469e-9),
    )