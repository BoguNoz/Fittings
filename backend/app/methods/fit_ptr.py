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
) -> PTRFitResult:
    """
    Main fitting routine for PTR amplitude and phase data.

    Automatically selects between 'deg' and 'rad' for phase if set to "auto".
    """
    # Initial guess and bounds [log10(k2), log10(alfa2), log10(r32), log10(k3), phi0_deg]
    p0 = np.array([np.log10(0.5), np.log10(3e-7), np.log10(1e-7), np.log10(3.0), 0.0])

    lb = np.array([np.log10(1e-3), np.log10(1e-10), np.log10(1e-10), np.log10(0.1), -360.0])
    ub = np.array([np.log10(1e2), np.log10(1e-4), np.log10(1e-4), np.log10(20.0), 360.0])

    if phase_units.lower() == "auto":
        res_deg = least_squares(ptr_residual, p0, bounds=(lb, ub),
                                args=(frequency_vector, exp_amp, exp_phase, "deg"))
        res_rad = least_squares(ptr_residual, p0, bounds=(lb, ub),
                                args=(frequency_vector, exp_amp, exp_phase, "rad"))

        res = res_deg if res_deg.cost <= res_rad.cost else res_rad
        used_units = "deg" if res_deg.cost <= res_rad.cost else "rad"
    else:
        res = least_squares(ptr_residual, p0, bounds=(lb, ub),
                            args=(frequency_vector, exp_amp, exp_phase, phase_units))
        used_units = phase_units.lower()

    pfit = res.x

    k2 = 10 ** pfit[0]
    alfa2 = 10 ** pfit[1]
    r32 = 10 ** pfit[2]
    k3 = 10 ** pfit[3]
    phi0_deg = pfit[4]

    # Compute best-fit model
    _, y_complex = simulations_ptr(frequency_vector, k2, alfa2, r32, k3)
    y_normalized = (y_complex / y_complex[0]) * np.exp(1j * phi0_deg * np.pi / 180.0)

    model_amp = np.abs(y_normalized)
    model_phase_deg = np.unwrap(np.angle(y_normalized)) * 180 / np.pi

    # Experimental phase for plotting
    if used_units == "deg":
        exp_phase_plot = exp_phase
    else:
        exp_phase_plot = exp_phase * 180 / np.pi

    exp_phase_plot = np.unwrap(exp_phase_plot * np.pi / 180) * 180 / np.pi

    return PTRFitResult(
        k2=k2,
        alfa2=alfa2,
        r32=r32,
        k3=k3,
        phi0_deg=phi0_deg,
        res_norm=2 * res.cost,
        model_amp=model_amp,
        model_phase_deg=model_phase_deg,
        exp_phase_deg=exp_phase_plot,
        phase_units=used_units,
        pfit=pfit,
        exit_flag=res.status,
        frequency_vector=frequency_vector  # optional
    )