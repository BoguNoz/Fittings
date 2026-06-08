from fastapi import APIRouter, UploadFile, File, Form
from joblib._multiprocessing_helpers import mp
from pydantic import BaseModel
from typing import List, Tuple
import numpy as np

from app.common.dto.DataRequestDto import DataRequestDto
from app.processors.ptr_processors_builder import FittingProcessorBuilder
from app.models.ptr_config import PTRConfig

router = APIRouter()

class PTRChartData(BaseModel):
    norm_amplitude_data: List[List[Tuple[float, float]]]
    phase_data: List[List[Tuple[float, float]]]
    results: dict


def format_for_echarts(x: np.ndarray, y: np.ndarray) -> List[Tuple[float, float]]:
    return np.column_stack((x, y)).tolist()



@router.post("/ptr-fitting", response_model=PTRChartData)
async def ptr_fitting(
    l1: float = Form(...),
    k1: float = Form(...),
    alfa1: float = Form(...),
    l2: float = Form(...),
    rhoc2: float = Form(...),
    alfa3: float = Form(...),
    k3: float = Form(...),
    r21: float = Form(...),
    d_pump: float = Form(...),
    Q: float = Form(...),
    anisotropy= Form(...),
    weight: float = Form(...),

    sample_name: str = Form(...),
    file: UploadFile = File(...),
    n_start: int = Form(...),
):
    config = PTRConfig(
        l1=l1,
        k1=k1,
        alfa1=alfa1,
        l2=l2,
        rhoc2=rhoc2,
        alfa3=alfa3,
        k3=k3,
        d_pump=d_pump,
        Q=Q,
        anisotropy=float(anisotropy),
        r21=r21,
        phase_weight=weight,
    )

    if __name__ == '__main__':
        mp.set_start_method('spawn', force=True)

    result = (FittingProcessorBuilder()
              .load_dat_data(file_source=file)
              .load_config(config)
              .set_starting_point_count(n_start)
              .build()
              .process())

    freq = result.frequency_hz
    freq_log = np.log10(freq)


    norm_amplitude_data = [
        format_for_echarts(freq_log, result.model_amp),
        format_for_echarts(freq_log, result.model_amp),
    ]

    phase_data = [
        format_for_echarts(freq_log, result.model_phase_deg),
        format_for_echarts(freq_log, result.exp_phase_deg),
    ]

    return PTRChartData(
        norm_amplitude_data=norm_amplitude_data,
        phase_data=phase_data,
        results={
            "anisotropy": result.anisotropy,
            "k2": float(result.k2),
            "alfa2": float(result.alfa2),
            "r32": float(result.r32),
            "kParallel": float(result.k_parallel),
            "r2Amp": float(result.r2_amp),
            "r2Phase": float(result.r2_phase),
            "res_norm": float(result.res_norm),
        }
    )