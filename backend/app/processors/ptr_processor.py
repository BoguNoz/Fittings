import numpy as np

from app.methods.mulitilayer.corrections import correct_ptr_data
from app.methods.mulitilayer.fitting import fit_ptr_3d
from app.methods.singlelayer.correct_tr import correct_tr_data
from app.methods.singlelayer.fitting_tr import fit_tr_1d
from app.models.ptr_config import PTRConfig
from app.models.ptr_data import PTRData
from app.models.ptr_fit_result import PTRFitResult


class PTRProcessor:
    def __init__(self):
        # Data
        self._data: PTRData = None
        self._config: PTRConfig = None
        self._phase_units: str = "rad"

        # Methods
        self._phase_correction: float = 0.0
        self._starting_points_count: int = 1


    def process(self) -> PTRFitResult:
        self.apply_phase_correction()
        return self.build_and_fit()

    def load_data(self, data: PTRData) -> 'PTRProcessor':
        self._data = data
        return self

    def set_config(self, config: PTRConfig) -> 'PTRProcessor':
        self._config = config
        return self

    def apply_phase_correction(self):
        return self

    def build_and_fit(self) -> PTRFitResult:
        """Build corrected data and run multi-start fitting."""

        freq_hz, amp, phase_deg = correct_ptr_data(
            self._data.frequency,
            self._data.amplitude,
            self._data.phase_deg
        )

        p0 = np.array([np.log10(8.0), np.log10(3.5e-6), np.log10(6.1e-8), 0.0, 0.0])

        return fit_ptr_3d(
            freq_hz=freq_hz,
            exp_amp=amp,
            exp_phase_deg=phase_deg,
            n_starts=self._starting_points_count,
            config=self._config,
            p0_expected=p0
        )

    def build_and_fit_tr(self) -> PTRFitResult:

        time_s, clean_signal = correct_tr_data(self._time_ns, self._signal)

        result = fit_tr_1d(
            time_s=time_s,
            exp_signal=clean_signal,
            config=self._config,
            n_starts=self._starting_points_count
        )