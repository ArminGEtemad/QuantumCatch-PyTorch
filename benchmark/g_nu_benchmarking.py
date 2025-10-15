import torch
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from spectra_calculator import liouvillian_superoperator, diag_liouvillian, g_nu, g_nu_batch


def benchmark_gnu_versions():
    H = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    c_ops = [torch.tensor([[1.0, 0.0], [0.0, 0.0]])]
    m_ops = [torch.tensor([[1.0, 0.0], [0.0, -1.0]])]
    gamma = [1.0]
    beta = [1.0]

    L = liouvillian_superoperator(H, c_ops, m_ops, gamma, beta)
    evals, V, V_inv, zero_ind, small_mask, diag = diag_liouvillian(L)

    # Frequency range
    nu_values = torch.linspace(-5.0, 5.0, 5000)

    t0 = time.perf_counter()
    G_loop = []
    for nu in nu_values:
        G = g_nu(evals, V, V_inv, zero_ind, small_mask, diag.clone(), nu)
        G_loop.append(G)
    G_loop = torch.stack(G_loop)
    t1 = time.perf_counter()
    loop_time = t1 - t0

    t2 = time.perf_counter()
    G_batch = g_nu_batch(evals, V, V_inv, zero_ind, small_mask, nu_values)
    t3 = time.perf_counter()
    batch_time = t3 - t2

    # Results
    print(f"Loop version time:  {loop_time:.6f} seconds")
    print(f"Batch version time: {batch_time:.6f} seconds")
    print(f"Speedup: {loop_time / batch_time:.2f}")

if __name__ == "__main__":
    benchmark_gnu_versions()
