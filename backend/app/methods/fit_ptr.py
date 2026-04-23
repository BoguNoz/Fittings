import numpy as np
from scipy.optimize import least_squares

from app.methods.ptr_residual import ptr_residual
from app.methods.simulations_ptr import simulations_ptr
from app.models.ptr_fit_result import PTRFitResult


def fit_ptr(
        frequency_vector: np.ndarray,
        exp_amp: np.ndarray,
        exp_phase: np.ndarray,
        phase_units: str = "auto",
        **phys_params
) -> PTRFitResult:
    """
    Main fitting routine for PTR amplitude and phase data.
    Automatically detects if experimental phase is in degrees or radians.
    """
    # Initial guess and bounds in log10 scale
    p0 = np.array([np.log10(0.5), np.log10(3e-7), np.log10(1e-7), np.log10(3.0), 0.0])
    lb = np.array([np.log10(1e-3), np.log10(1e-10), np.log10(1e-10), np.log10(0.1), -360.0])
    ub = np.array([np.log10(1e2), np.log10(1e-4), np.log10(1e-4), np.log10(20.0), 360.0])

    # Run optimization for both degree and radian assumptions if "auto"
    if phase_units.lower() == "auto":
        res_deg = least_squares(ptr_residual, p0, bounds=(lb, ub),
                                args=(frequency_vector, exp_amp, exp_phase, "deg"),
                                kwargs=phys_params)
        res_rad = least_squares(ptr_residual, p0, bounds=(lb, ub),
                                args=(frequency_vector, exp_amp, exp_phase, "rad"),
                                kwargs=phys_params)

        if res_deg.cost <= res_rad.cost:
            res, used_units = res_deg, "deg"
        else:
            res, used_units = res_rad, "rad"
    else:
        used_units = phase_units.lower()
        res = least_squares(ptr_residual, p0, bounds=(lb, ub),
                            args=(frequency_vector, exp_amp, exp_phase, used_units),
                            kwargs=phys_params)

    # Convert fitted parameters back to physical scale
    pfit = res.x
    k2, alfa2, r32, k3, phi0_deg = 10 ** pfit[0], 10 ** pfit[1], 10 ** pfit[2], 10 ** pfit[3], pfit[4]

    # Generate final model curves
    _, y_complex = simulations_ptr(frequency_vector, k2, alfa2, r32, k3, **phys_params)
    y_norm = (y_complex / y_complex[0]) * np.exp(1j * np.deg2rad(phi0_deg))

    model_amp = np.abs(y_norm)
    model_phase_deg = np.unwrap(np.angle(y_norm)) * 180 / np.pi

    # Prepare experimental phase for consistent plotting
    exp_phase_deg = exp_phase if used_units == "deg" else np.rad2deg(exp_phase)
    exp_phase_deg_plot = np.unwrap(np.deg2rad(exp_phase_deg)) * 180 / np.pi

    return PTRFitResult(
        k2=k2,
        alfa2=alfa2,
        r32=r32,
        k3=k3,
        phi0_deg=phi0_deg,
        res_norm=2 * res.cost,
        model_amp=model_amp,
        model_phase_deg=model_phase_deg,
        exp_phase_deg=exp_phase_deg_plot,
        phase_units=used_units,
        pfit=pfit,
        exit_flag=res.status,
        frequency_vector=frequency_vector,
        l2=phys_params.get('l2', 469e-9)
    )