import torch
from multichqc.system_config import SystemConfig
from multichqc.input_config import SpectrumConfig
from multichqc.spectra_calculator import SpectrumCalculator
from multichqc.plot_config import PlotConfig
from multichqc.spectra_plotter import SpectrumPlotter

#----- EXAMPLE SYSTEM ------
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
beta_dict = {'0': 5., '1': 3.}

measure_order = {'s3': [('1', '0', '0')]}

#----- SPECTRUM CONFIG -----
resolution = 500
f_min = 0
f_max = 2.0

spe_config = SpectrumConfig(
    resolution=resolution,
    f_min=f_min,
    f_max=f_max,
    order_combi=measure_order,
    backend='cpu'
)

sys_config = SystemConfig(
    hamiltonian=H,
    collapse_ops=c_ops,
    measurement_ops=measure_ops,
    collapse_strength=gamma_list,
    measurement_strength=beta_dict
)

#----- SPECTRUM CALCULATION -----
calc = SpectrumCalculator(spe_config, sys_config)
import matplotlib.pyplot as plt

s3 = calc.s3[('1', '0', '0')]

f_min = spe_config.f_min
f_max = spe_config.f_max
extent = (f_min, f_max, f_min, f_max)

plt.figure(figsize=(6, 5))
plt.imshow(s3.real, extent=extent, origin='lower', cmap='seismic', aspect='auto')
plt.colorbar(label='S3 Amplitude')
plt.title("S3 Spectrum for ('1','0','0')")
plt.xlabel('Frequency $f_1$')
plt.ylabel('Frequency $f_2$')
plt.tight_layout()
plt.show()

