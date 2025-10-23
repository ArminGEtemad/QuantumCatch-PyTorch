import torch

from multichqc.spectra_calculator import liouvillian_superoperator, rho_steady

H = torch.tensor([[1., 0.], [0., -1.]]) * 0

c_ops = [
    torch.tensor([[0., 1.], [0., 0.]]),
    torch.tensor([[0., 0.], [1., 0.]])
]
gamma_list = torch.tensor([0.3, 0.5])

measure_ops = {
    '0': torch.tensor([[0., 0.], [0., 1.]])
}
beta_dict = {'0': 5.}

L = liouvillian_superoperator(
        H, c_ops, measure_ops,
        gamma_list, beta_dict, 'mps'
    )
rho = rho_steady(H, L, 'mps')

print(L)
print(rho)