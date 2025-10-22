from dataclasses import dataclass
from typing import Sequence, Dict
from torch import Tensor

@dataclass
class SystemConfig:
    hamiltonian: Tensor
    collapse_ops: Sequence[Tensor]
    collapse_strength: Tensor
    measurement_ops: Dict[str, Tensor]
    measurement_strength: Dict[str, float]

    
