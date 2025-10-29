import torch
import time
from statistics import mean

from multichqc.system_config import SystemConfig
from multichqc.input_config import SpectrumConfig
from multichqc.spectra_calculator import SpectrumCalculator
from multichqc.plot_config import PlotConfig
from multichqc.spectra_plotter import SpectrumPlotter

# ----- FIXED SYSTEM -----
H = torch.tensor([[1., 0.], [0., -1.]]) * 0
c_ops = [
    torch.tensor([[0., 1.], [0., 0.]]),
    torch.tensor([[0., 0.], [1., 0.]])
]
gamma_list = torch.tensor([0.3, 0.5])
measure_ops = {
    '0': torch.tensor([[0., 0.], [0., 1.]]),
    '1': torch.tensor([[1., 0.], [0., 0.]])
}
beta_dict = {'0': 5., '1': 1.}
measure_order = {
    'ss': [('0', '0')],
    's3': [('0', '0', '0')],
    's4': [('0', '0', '0', '0')]
}
f_min = -1.0
f_max = 1.0
backend = 'cpu'

# ----- RESOLUTIONS TO TEST -----
resolutions = [2**i for i in range(5, 13)]

# ----- BENCHMARK LOOP -----
print(f"{'Resolution':>10} | {'Avg Time (s)':>14}")
print("-" * 30)

for res in resolutions:
    durations = []
    for _ in range(3):
        spe_config = SpectrumConfig(
            resolution=res,
            f_min=f_min,
            f_max=f_max,
            order_combi=measure_order,
            s3_block=None,
            s4_block=None,
            backend=backend,
        )

        sys_config = SystemConfig(
            hamiltonian=H,
            collapse_ops=c_ops,
            measurement_ops=measure_ops,
            collapse_strength=gamma_list,
            measurement_strength=beta_dict
        )

        # Warm-up
        _ = SpectrumCalculator(spe_config, sys_config)

        # Timed run
        t0 = time.perf_counter()
        _ = SpectrumCalculator(spe_config, sys_config)
        t1 = time.perf_counter()

        durations.append(t1 - t0)

    avg_time = mean(durations)
    print(f"{res:>10} | {avg_time:>14.6f}")
