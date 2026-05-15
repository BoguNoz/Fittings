from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Tuple
import numpy as np

from app.common.dto.DataRequestDto import DataRequestDto
from app.fitting_processors.ptr_processors_builder import FittingProcessorBuilder
from app.models.ptr_config import PTRConfig

router = APIRouter()

class PTRChartData(BaseModel):
    amplitude_data: List[List[Tuple[float, float]]]
    norm_amplitude_data: List[List[Tuple[float, float]]]
    phase_data: List[List[Tuple[float, float]]]
    results: dict


def format_for_echarts(x: np.ndarray, y: np.ndarray) -> List[Tuple[float, float]]:
    return np.column_stack((x, y)).tolist()


@router.post("/ptr-fitting", response_model=PTRChartData)
async def ptr_fitting(
    l2: float = Form(...),
    k1: float = Form(...),
    l1: float = Form(...),
    alfa1: float = Form(...),
    alfa2: float = Form(...),
    alfa3: float = Form(...),
    r21: float = Form(...),
    weight: float = Form(...),

    sample_name: str = Form(...),
    use_hankel: bool = Form(...),

    file: UploadFile = File(...),
):
    config = PTRConfig(
        l2=l2,
        k1=k1,
        l1=l1,
        alfa1=alfa1,
        alfa2=alfa2,
        alfa3=alfa3,
        r21=r21,
        weight=weight,
    )

    result = (FittingProcessorBuilder()
              .load_dat_data(file_source=file, sample_name=sample_name)
              .load_config(config)
              .build()
              .process())

    # Reszta kodu bez zmian...
    freq = result.frequency_vector
    freq_log = np.log10(freq)

    amplitude_data = [
        format_for_echarts(freq, result.model_amp),
        format_for_echarts(freq, result.exp_amp),
    ]

    norm_amp_model = result.model_amp / result.model_amp[0]
    norm_amp_exp = result.exp_amp / result.exp_amp[0]

    norm_amplitude_data = [
        format_for_echarts(freq_log, np.log10(norm_amp_model)),
        format_for_echarts(freq_log, np.log10(norm_amp_exp)),
    ]

    phase_data = [
        format_for_echarts(freq_log, result.model_phase_deg),
        format_for_echarts(freq_log, result.exp_phase_deg),
    ]

    return PTRChartData(
        amplitude_data=amplitude_data,
        norm_amplitude_data=norm_amplitude_data,
        phase_data=phase_data,
        results={
            "k2": float(result.k2),
            "alfa2r": float(result.alfa2),
            "r32": float(result.r32),
            "phi0_deg": float(result.phi0_deg),
            "res_norm": float(result.res_norm),
        }
    )