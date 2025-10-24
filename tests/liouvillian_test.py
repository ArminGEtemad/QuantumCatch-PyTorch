import torch
import numpy as np
import pytest

from multichqc.spectra_calculator import liouvillian_superoperator


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
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_random_liouvillian_matches_reference(seed, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA NOT AVAILABLE.")
    
    np.random.seed(seed)

    dim = 3
    num_collapse = 3
    num_measure = 3

    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    H_np = (A + A.conj().T) / 2

    c_ops_np = [np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim) for _ in range(num_collapse)]
    m_ops_np = [np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim) for _ in range(num_measure)]

    gamma = np.random.rand(num_collapse)
    beta  = np.random.rand(num_measure)

    ref_ops_np = [np.sqrt(gamma[k]) * c_ops_np[k] for k in range(num_collapse)] + \
                 [beta[k] * m_ops_np[k] for k in range(num_measure)]

    L_np = calc_super_liou(H_np, ref_ops_np)

    H_torch = torch.tensor(H_np, dtype=torch.cfloat)
    c_ops_torch = [torch.tensor(op, dtype=torch.cfloat) for op in c_ops_np]
    m_ops_torch = [torch.tensor(op, dtype=torch.cfloat) for op in m_ops_np]

    # ✅ Convert to dicts
    m_ops_dict = {str(k): m_ops_torch[k] for k in range(num_measure)}
    beta_dict  = {str(k): beta[k] for k in range(num_measure)}

    L_torch = liouvillian_superoperator(
        H_torch, c_ops_torch, m_ops_dict,
        gamma.tolist(), beta_dict, device
    ).cpu().numpy()

    np.testing.assert_allclose(L_torch, L_np, atol=1e-6, rtol=1e-5)