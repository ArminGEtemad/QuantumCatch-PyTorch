from qutip import jmat, tensor, qeye
import torch
import numpy as np
from multichqc.system_config import SystemConfig
from multichqc.input_config import SpectrumConfig
from multichqc.spectra_calculator import SpectrumCalculator
from multichqc.plot_config import PlotConfig
from multichqc.spectra_plotter import SpectrumPlotter

# 2) Physical constants & parameters
hbar = 6.582e-4    # eV·ps
h = 2 * np.pi * hbar

B0 = 0.01
B_vec = B0 * np.array([1, 0, 0])

beta_g_e = 0.172 * hbar
beta_g_n = 9.329e-6 * hbar
A = 100.2e-6 * h
P = 1.27e-6 * h

# 3) Build Qutip spin operators
n_e_states = 2
n_n_states = 10
n_e = (n_e_states - 1) / 2
n_n = (n_n_states - 1) / 2

sx_e = jmat(n_e, 'x'); sy_e = jmat(n_e, 'y'); sz_e = jmat(n_e, 'z')
sx_n = jmat(n_n, 'x'); sy_n = jmat(n_n, 'y'); sz_n = jmat(n_n, 'z')

# 4) Hamiltonian in Qutip, then to NumPy
s_e = [sx_e, sy_e, sz_e]
s_n = [sx_n, sy_n, sz_n]

H_q = (
    beta_g_e * tensor(sum(B_vec[i]*s_e[i] for i in range(3)), qeye(n_n_states))
  + A      * sum(tensor(s_e[i], s_n[i]) for i in range(3))
  + P      * tensor(qeye(n_e_states), sz_n**2)
  - beta_g_n * tensor(qeye(n_e_states), sum(B_vec[i]*s_n[i] for i in range(3)))
)
H_np = H_q.full() / hbar


H = torch.tensor(H_np)

d1 = torch.tensor(tensor(qeye(n_e_states), sz_n).full(), dtype=torch.cfloat)
d2 = torch.tensor(tensor(sz_e, qeye(n_n_states)).full(), dtype=torch.cfloat)
c_ops = [d1, d2]
gamma = torch.tensor([5e-8, 5e-5], dtype=torch.cfloat)

c1 = torch.tensor(tensor(sz_e, qeye(n_n_states)).full(), dtype=torch.cfloat)
c2 = torch.tensor(tensor(qeye(n_e_states), sx_n).full(), dtype=torch.cfloat)


measure_ops = {
    'z_ele' : c1,
    'x_nuc' : c2,
}

beta  = {
    'z_ele' : 1e-2,
    'x_nuc' : 1e-2,
}

# 8) Frequency grid
resolution = 500
f_min, f_max = -0.001, 0.001
f = torch.linspace(f_min, f_max, resolution)

#-----------------------------------------------------------------------------------------------------------


spe_config = SpectrumConfig(
    resolution=resolution,
    f_min=f_min,
    f_max=f_max,
    order_combi={'s4': [('x_nuc', 'x_nuc', 'z_ele', 'z_ele')]},
    s3_block=300,
    s4_block=128,
    backend='cuda'
)
sys_config = SystemConfig(
    hamiltonian=H,
    collapse_ops=c_ops,
    measurement_ops=measure_ops,
    collapse_strength=gamma,
    measurement_strength=beta
)
calc = SpectrumCalculator(spe_config, sys_config)

plot_config = PlotConfig(
    f_min=-0.001,
    f_max=0.001,
    display_orders=[4],
    plot_format=["re", "im"],
)

plotter = SpectrumPlotter(calc, plot_config)
plotter.display()