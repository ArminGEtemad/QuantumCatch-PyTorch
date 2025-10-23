import torch
import matplotlib.pyplot as plt
from torch import Tensor
from typing import Dict

from multichqc.spectra_calculator import liouvillian_superoperator, rho_steady, mathcal_a, mathcal_a_prim, diag_liouvillian, g_nu_batch, freq_to_circfreq, g_nu_batch_unique, s2_batched
COMPLEX64_TYPE = torch.cfloat

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

f = torch.linspace(-2, 2, 200)
measure_order = {'s2': [('0', '0'), ('1', '1'), ('1', '0')]}

backend = 'cuda'
# ---------------------------------------------------
omega = freq_to_circfreq(f, backend)

L = liouvillian_superoperator(
        H, c_ops, measure_ops,
        gamma_list, beta_dict, backend
    )
rho = rho_steady(H, L, backend)

A = mathcal_a(H, measure_ops, backend)
A_prim = mathcal_a_prim(H, rho, measure_ops, backend)
D_L = diag_liouvillian(L)
g_nu = g_nu_batch(D_L[0], D_L[1], D_L[2], D_L[3], D_L[4], omega)
g_nu_un = g_nu_batch_unique(D_L[0], D_L[1], D_L[2], D_L[3], D_L[4], omega)

s2 = s2_batched(H, L, measure_ops, beta_dict, measure_order , rho, omega, backend , key='s2')
print(L)
print(rho)
print(A)
print(A_prim)
print(D_L)
print(g_nu)
print(g_nu_un)

plt.plot(s2[('1', '1')].real)
plt.show()