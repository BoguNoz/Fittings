import numpy as np

from app.methods.corrections import correct_ptr_data
from app.methods.fitting import fit_ptr_multi_start
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
        # 1. Preprocessing
        freq, amp, phase = correct_ptr_data(
            self._data.frequency,
            self._data.amplitude,
            self._data.phase_deg,
            True
        )

        # 2. Scale frequency to Hz (as expected by fitting functions)
        freq_hz = freq * 1000

        # 3. Run fitting
        return fit_ptr_multi_start(
            frequency_vector=freq_hz,  # teraz w Hz
            exp_amp=amp,  # bez mnożenia przez 1000
            exp_phase=phase,
            n_starts=self._starting_points_count,

            # Parametry z config
            l2=self._config.l2,
            k1=self._config.k1,
            l1=self._config.l1,
            alfa1=self._config.alfa1,
            alfa3=self._config.alfa3,
            r21=self._config.r21,
            rhoc=self._config.rhoc,
            d_pump=self._config.d_pump,
            Q=self._config.Q,
            weight_exponent=self._config.weight_exponent,
            phase_weight=self._config.phase_weight,
        )
