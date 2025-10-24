import numpy as np
import torch
import pytest
from multichqc.spectra_calculator import mathcal_a 


def calc_super_A(op):
    def calc_A(rho, _op):
        return (_op @ rho + rho @ np.conj(_op).T) / 2

    m, n = op.shape
    op_super = 1j * np.ones((n ** 2, n ** 2), dtype=complex)
    for j in range(n ** 2):
        rho_vec = np.zeros(n ** 2, dtype=complex)
        rho_vec[j] = 1
        rho_mat = rho_vec.reshape((m, n))
        rho_dot = calc_A(rho_mat, op)
        rho_dot = rho_dot.reshape((n ** 2))
        op_super[:, j] = rho_dot
    return op_super

"""
For the old code:
Had to change the Transpose and not multiply with the measurement 
strength for the test to pass. (fixing an error in the old code)
"""
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mathcal_a_matches(seed, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA NOT AVAILABLE.")
    
    np.random.seed(seed)

    n = 3

    # Random Hermitian Hamiltonian
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    H_np = (A + A.conj().T) / 2

    # Random measurement operator
    op_np = np.random.randn(n, n) + 1j * np.random.randn(n, n)

    H_torch = torch.tensor(H_np, dtype=torch.cfloat)
    op_torch = torch.tensor(op_np, dtype=torch.cfloat)

    # 🛠 Use a dict to match the new function signature
    A_dict_torch = mathcal_a(H_torch, {'0': op_torch}, device)
    A_torch = A_dict_torch['0'].cpu().numpy()

    # Reference calculation
    A_base = calc_super_A(op_np)

    # Assertion
    np.testing.assert_allclose(A_torch, A_base, rtol=1e-5, atol=1e-6)