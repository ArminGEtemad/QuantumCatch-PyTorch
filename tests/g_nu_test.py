import torch
import numpy as np
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from spectra_calculator import liouvillian_superoperator, diag_liouvillian, g_nu


# Example Hamiltonian
H = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

# One collapse operator and one measurement operator (Pauli Z)
cOps = [torch.tensor([[1.0, 0.0], [0.0, 0.0]])]
mOps = [torch.tensor([[1.0, 0.0], [0.0, -1.0]])]
gamma = [1.0]
beta = [1.0]

L = liouvillian_superoperator(H, cOps, mOps, gamma, beta)
evals, evecs, evecs_inv, zero_ind, small_mask, diag = diag_liouvillian(L)

nu = torch.tensor(1.0)
G = g_nu(evals, evecs, evecs_inv, zero_ind, small_mask, diag, nu)

print("G(nu=1.0):")
print(G)
