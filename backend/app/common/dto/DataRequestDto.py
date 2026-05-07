from pathlib import Path
from typing import Union, BinaryIO

from pydantic import BaseModel, Field, ConfigDict

class DataRequestDto(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    l2: float
    k1: float
    l1: float
    alfa1: float
    alfa2: float
    alfa3: float
    r21: float

    sample_name: str
    use_hankel: bool
    file: Union[str, Path, BinaryIO]