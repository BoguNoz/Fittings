from pathlib import Path
from typing import Union, BinaryIO

import numpy as np

from app.processors.ptr_processor import PTRProcessor
from app.models.ptr_config import PTRConfig
from app.models.ptr_data import PTRData


class FittingProcessorBuilder:
    def __init__(self):
        self._processor = PTRProcessor()

    def build(self) -> PTRProcessor:
        return self._processor

    def load_dat_file(self, file_path: str, sample_name: str = "test") -> 'FittingProcessorBuilder':
        file_path = Path(file_path)

        # wczytaj tekst i zamień przecinki dziesiętne na kropki
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().replace(',', '.')

        from io import StringIO

        data = np.loadtxt(
            StringIO(content),
            skiprows=0,
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
        )

        self._processor.load_data(data)
        return self

    def load_dat_data(
            self,
            file_source,
    ) -> 'FittingProcessorBuilder':

        try:
            source = file_source.file if hasattr(file_source, "file") else file_source

            data = np.loadtxt(
                source,
                skiprows=0,
                delimiter=None,
                comments='#',
                usecols=(0, 1, 2)
            )

        except Exception as e:
            raise ValueError(f"File error: {e}")

        ptr_data = PTRData(
            frequency=data[:, 0],
            amplitude=data[:, 1],
            phase_deg=data[:, 2],
        )

        self._processor.load_data(ptr_data)

        return self

    def set_starting_point_count(self, count: int) -> 'FittingProcessorBuilder':
        self._processor._starting_points_count = count
        return self

    def load_config(self, config: PTRConfig) -> 'FittingProcessorBuilder':
        self._processor.set_config(config)
        return self

    def apply_phase_correction(self, delta_deg: float) -> 'FittingProcessorBuilder':
        self._processor._phase_correction = delta_deg
        return self

