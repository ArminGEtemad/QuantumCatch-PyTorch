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

measure_order = {'s2': [('0', '0'), ('1', '1'), ('1', '0')]}

#----- SPECTRUM CONFIG -----
resolution = 300
f_min = -2.0
f_max = 2.0

spe_config = SpectrumConfig(
    resolution=resolution,
    f_min=f_min,
    f_max=f_max,
    order_combi=measure_order,
    s3_block=None,
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

#----- PLOT CONFIG & PLOTTING -----
plot_config = PlotConfig(
    f_min=-2.0,
    f_max=2.0,
    display_orders=[2],
    plot_format=["re", "im"],
)

plotter = SpectrumPlotter(calc, plot_config)
plotter.display()
