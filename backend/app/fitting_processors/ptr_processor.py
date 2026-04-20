from app.methods.fit_ptr import fit_ptr
from app.models.ptr_config import PTRConfig
from app.models.ptr_data import PTRData
from app.models.ptr_fit_result import PTRFitResult


class PTRProcessor:
    def __init__(self):
        # Data
        self._data: PTRData = None
        self._config: PTRConfig = None
        self._phase_units: str = "auto"


    def process(self) -> PTRFitResult:
      return self.build_and_fit()

    def load_data(self, data: PTRData) -> 'PTRProcessor':
        self._data = data
        return self

    def set_config(self, config: PTRConfig) -> 'PTRProcessor':
        self._config = config
        return self

    def set_phase_units(self, units: str) -> 'PTRProcessor':
        if units not in ("auto", "deg", "rad"):
            raise ValueError(f"Invalid phase_units: {units}. Must be 'auto', 'deg' or 'rad'.")
        self._phase_units = units
        return self

    def build_and_fit(self) -> PTRFitResult:

        result: PTRFitResult = fit_ptr(
            frequency_vector=self._data.frequency,
            exp_amp=self._data.amplitude,
            exp_phase=self._data.phase_deg,
            phase_units=self._phase_units,
            l2=self._config.l2,
            k1=self._config.k1,
            l1=self._config.l1,
            alfa1=self._config.alfa1,
            alfa3=self._config.alfa3,
            r21=self._config.r21,
        )

        result.sample_name = self._data.sample_name
        return result

