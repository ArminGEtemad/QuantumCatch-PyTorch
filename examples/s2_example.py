# run_test_s2.py
import torch
import matplotlib.pyplot as plt

from multichqc.system_config import SystemConfig
from multichqc.input_config import SpectrumConfig
from multichqc.spectra_calculator import SpectrumCalculator


#----- EXAMPLE SYSTEM ------
H = torch.tensor([[1., 0.], [0., -1.]]) * 0

c_ops = [
    torch.tensor([[0., 1.], [0., 0.]]),
    torch.tensor([[0., 0.], [1., 0.]])
]
gamma_list = torch.tensor([0.3, 0.5])

measure_ops = {'0': torch.tensor([[0., 0.], [0., 1.]])}
beta_dict = {'0': 5.0}
measure_order = {'s2': [('0', '0')], 's3': [('1', '0')]}

#----- SPEC CONFIG -----
resolution = 300
f_min = -2.0
f_max = 2.0

#----- CONFIGS -----
sys_config = SystemConfig(
    hamiltonian=H,
    collapse_ops=c_ops,
    measurement_ops=measure_ops,
    collapse_strength=gamma_list,
    measurement_strength=beta_dict
)

spe_config = SpectrumConfig(
    resolution=resolution,
    f_min=f_min,
    f_max=f_max,
    order_combi=measure_order
)

#--- CALCULATION ---
calc = SpectrumCalculator(spe_config, sys_config)

if calc.s2 is not None:
    for (op0, op1), spectrum in calc.s2.items():
        plt.plot(calc.freq.numpy(), spectrum.real.numpy(), label=f"S2({op0}, {op1})")
    plt.xlabel("Frequency")
    plt.ylabel("S2")
    plt.title("S2")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("S₂ was not computed.")
