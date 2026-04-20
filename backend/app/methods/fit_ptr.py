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

    l2: float = 469e-9,
    k1: float = 21.0,
    l1: float = 80e-9,
    alfa1: float = 8.9e-6,
    alfa3: float = 6.0e-6,
    r21: float = 2.8e-8,
) -> PTRFitResult:
    """
    Main fitting routine for Photothermal Radiometry (PTR) amplitude and phase data.

    Automatically selects best phase units ('deg' or 'rad') when phase_units="auto".

    Parameters
    ----------
    frequency_vector : np.ndarray
        Frequencies in Hz.
    exp_amp : np.ndarray
        Experimental amplitude.
    exp_phase : np.ndarray
        Experimental phase.
    phase_units : str, default="auto"
        'deg', 'rad' or 'auto'.
    l2 : float, default=469e-9
        Thickness of the main ZnO film (Layer 2). Should come from RBS fitting_processors.

    Returns
    -------
    PTRFitResult
        Dataclass containing fitted parameters and model curves.
    """
    # Initial guess and bounds [log10(k2), log10(alfa2), log10(r32), log10(k3), phi0_deg]
    p0 = np.array([np.log10(0.5), np.log10(3e-7), np.log10(1e-7), np.log10(3.0), 0.0])

    lb = np.array([np.log10(1e-3), np.log10(1e-10), np.log10(1e-10), np.log10(0.1), -360.0])
    ub = np.array([np.log10(1e2),  np.log10(1e-4), np.log10(1e-4), np.log10(20.0),  360.0])

    if phase_units.lower() == "auto":
        res_deg = least_squares(ptr_residual, p0, bounds=(lb, ub),
                                args=(frequency_vector, exp_amp, exp_phase, "deg"))
        res_rad = least_squares(ptr_residual, p0, bounds=(lb, ub),
                                args=(frequency_vector, exp_amp, exp_phase, "rad"))

        res = res_deg if res_deg.cost <= res_rad.cost else res_rad
        used_units = "deg" if res_deg.cost <= res_rad.cost else "rad"
    else:
        res = least_squares(ptr_residual, p0, bounds=(lb, ub),
                            args=(frequency_vector, exp_amp, exp_phase, phase_units.lower()))
        used_units = phase_units.lower()

    pfit = res.x

    # Convert back to physical units
    k2 = 10 ** pfit[0]
    alfa2 = 10 ** pfit[1]
    r32 = 10 ** pfit[2]
    k3 = 10 ** pfit[3]
    phi0_deg = pfit[4]

    # Compute best-fit model
    _, y_complex = simulations_ptr(frequency_vector, k2, alfa2, r32, k3, l2=l2, k1=k1, alfa1=alfa1, alfa3=alfa3, l1=l1, r21=r21)

    y_normalized = (y_complex / y_complex[0]) * np.exp(1j * np.deg2rad(phi0_deg))

    model_amp = np.abs(y_normalized)
    model_phase_deg = np.unwrap(np.angle(y_normalized)) * 180 / np.pi

    # Prepare experimental phase for consistent plotting
    if used_units == "deg":
        exp_phase_plot = exp_phase
    else:
        exp_phase_plot = np.rad2deg(exp_phase)

    exp_phase_plot = np.unwrap(np.deg2rad(exp_phase_plot)) * 180 / np.pi

    return PTRFitResult(
        k2=k2,
        alfa2=alfa2,
        r32=r32,
        k3=k3,
        phi0_deg=phi0_deg,
        res_norm=2 * res.cost,                    # sum of squared weighted residuals
        model_amp=model_amp,
        model_phase_deg=model_phase_deg,
        exp_phase_deg=exp_phase_plot,
        phase_units=used_units,
        pfit=pfit,
        exit_flag=res.status,
        frequency_vector=frequency_vector,
        l2=l2                                     # save used thickness
    )