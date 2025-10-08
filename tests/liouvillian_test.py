import torch
import numpy as np
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from spectra_calculator import liouvillian_superoperator


# Reference Liouvillian from the old version

def cmtr(a, b):
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    return a @ b - b @ a

def calc_liou(rho_, h, c_ops_):
    rho_ = np.ascontiguousarray(rho_)
    liou = 1j * cmtr(rho_, h)
    for c_op in c_ops_:
        c_op = np.ascontiguousarray(c_op)
        liou += c_op @ rho_ @ c_op.conj().T - \
                0.5 * (c_op.conj().T @ c_op @ rho_ + rho_ @ c_op.conj().T @ c_op)
    return liou

def calc_super_liou(h_, c_ops):
    m, n = h_.shape
    op_super = 1j * np.ones((n ** 2, n ** 2), dtype=complex)
    for j in range(n ** 2):
        rho_vec = 1j * np.zeros(n ** 2, dtype=complex)
        rho_vec[j] = 1
        rho_mat = rho_vec.reshape((m, n))
        rho_dot = calc_liou(rho_mat, h_, c_ops)
        rho_dot = rho_dot.reshape((n ** 2))
        op_super[:, j] = rho_dot
    return op_super

# Test
def test_liouvillian_superoperator_matches_reference():
    # PyTorch inputs
    h = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.cfloat)
    c_ops = [torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.cfloat)]
    m_ops = [torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.cfloat)]

    gamma = [1.0]
    beta = [1.0]

    L_torch = liouvillian_superoperator(h, c_ops, m_ops, gamma, beta) # pytorch implementation
    L_np = calc_super_liou(
        h.numpy(), 
        [op.numpy() for op in c_ops + m_ops]
    ) # old algorithm

    assert L_torch.shape == L_np.shape
    np.testing.assert_allclose(L_torch.numpy(), L_np, atol=1e-6, rtol=1e-5)
