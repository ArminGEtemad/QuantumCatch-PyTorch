from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class SpectrumConfig:
    resolution: int
    f_min: float
    f_max: float
    order_combi: Dict[str, list[Tuple[str, str]]]

    def __post_init__(self):
        if self.f_min >= self.f_max:
            raise ValueError("f_min must be less than f_max.")
        if self.resolution <= 0:
            raise ValueError("resolution must be a positive integer.")
