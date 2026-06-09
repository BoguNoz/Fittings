from app.methods.corrections import correct_ptr_data
from app.methods.mulitilayer.fitting import fit_ptr_3d
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

        return fit_ptr_3d(
            freq_hz=freq_hz,
            exp_amp=amp,
            exp_phase_deg=phase_deg,
            n_starts=self._starting_points_count,
            config=self._config
        )