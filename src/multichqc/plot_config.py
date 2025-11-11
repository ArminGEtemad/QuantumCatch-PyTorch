# Copyright 2025 Armin Ghorbanietemad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Sequence, Literal

@dataclass
class PlotConfig:
    f_min: float
    f_max: float
    display_orders: Sequence[int]
    plot_format: Sequence[Literal["re", "im"]] = ("re",)