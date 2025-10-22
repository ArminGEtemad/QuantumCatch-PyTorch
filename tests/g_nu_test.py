import torch
import numpy as np

import pytest

from multichqc.spectra_calculator import liouvillian_superoperator, diag_liouvillian, g_nu_batch, g_nu_batch_unique

def g_nu(evals, evecs, evecs_inverse, zero_ind, small_indices, diagonal, nu):
    diagonal[~small_indices] = 1 / (-evals[~small_indices] - 1j * nu)
    diagonal[zero_ind] = 0

    Fourier_G = evecs @ torch.diag(diagonal) @ evecs_inverse

    return Fourier_G


@pytest.mark.parametrize("device", ["cpu"])
def test_g_nu_vs_batched(device):
    # example system
    H = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    c_ops = [torch.tensor([[1.0, 0.0], [0.0, 0.0]])]
    m_ops = {'0': torch.tensor([[1.0, 0.0], [0.0, -1.0]])}  # Pauli-Z
    gamma = [1.0]
    beta = {'0': 1.0}

    L = liouvillian_superoperator(H, c_ops, m_ops, gamma, beta)
    evals, V, V_inv, zero_ind, small_mask, diag = diag_liouvillian(L)

    nu_values = torch.linspace(-1.0, 1.0, 100).to(device)
    G_batched = g_nu_batch(evals, V, V_inv, zero_ind, small_mask, nu_values)

    G_loop = []
    for nu in nu_values:
        G = g_nu(evals, V, V_inv, zero_ind, small_mask, diag.clone(), nu)
        G_loop.append(G)

    G_loop = torch.stack(G_loop)

    # test
    torch.testing.assert_close(
        G_batched, G_loop, rtol=1e-5, atol=1e-7, msg="Mismatch between g_nu and g_nu_batch!"
    )

@pytest.mark.parametrize("device", ["cpu"])
def test_g_nu_batch_unique_equivalence(device):
    H = torch.tensor([[0.0, 1.0], [1.0, 0.0]]).to(device)
    c_ops = [torch.tensor([[1.0, 0.0], [0.0, 0.0]]).to(device)]
    m_ops = {'0': torch.tensor([[1.0, 0.0], [0.0, -1.0]]).to(device)}
    gamma = [1.0]
    beta = {'0': 1.0}

    L = liouvillian_superoperator(H, c_ops, m_ops, gamma, beta)
    evals, V, V_inv, zero_ind, small_mask, diag = diag_liouvillian(L)

    nu_values = torch.linspace(-5.0, 5.0, 500, device=device)
    G_regular = g_nu_batch(evals, V, V_inv, zero_ind, small_mask, nu_values)
    G_unique  = g_nu_batch_unique(evals, V, V_inv, zero_ind, small_mask, nu_values)

    torch.testing.assert_close(
        G_regular, G_unique, rtol=1e-6, atol=1e-8,
        msg="Mismatch between g_nu_batch and g_nu_batch_unique for unique ν values."
    )