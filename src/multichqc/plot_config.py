from dataclasses import dataclass
from typing import Sequence, Literal

@dataclass
class PlotConfig:
    f_min: float
    f_max: float
    display_orders: Sequence[int]
    plot_format: Sequence[Literal["re", "im"]] = ("re",)