from pathlib import Path

import numpy as np

from app.fitting_processors.ptr_processor import PTRProcessor
from app.models.ptr_config import PTRConfig
from app.models.ptr_data import PTRData


class FittingProcessorBuilder:
    def __init__(self):
        self._processor = PTRProcessor()

    def build(self) -> PTRProcessor:
        return self._processor

    def load_dat_file(self, file_path: str, sample_name: str = "test") -> 'FittingProcessorBuilder':
        file_path = Path(file_path)
        data = np.loadtxt(
            file_path,
            skiprows=0,
            delimiter=None,
            comments='#',
            usecols=(0, 1, 2)
        )

        frequency = data[:, 0]
        amplitude = data[:, 1]
        phase_deg = data[:, 2]

        data = PTRData(
            frequency=frequency,
            amplitude=amplitude,
            phase_deg=phase_deg,
            sample_name=sample_name
        )

        self._processor.load_data(data)
        return self

    def load_config(self, config: PTRConfig) -> 'FittingProcessorBuilder':
        self._processor.set_config(config)
        return self

    def apply_phase_correction(self, delta_deg: float) -> 'FittingProcessorBuilder':
        self._processor._phase_correction = delta_deg
        return self

