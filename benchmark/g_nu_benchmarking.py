import torch
import time
from multichqc.spectra_calculator import liouvillian_superoperator, diag_liouvillian, g_nu_batch


def g_nu(evals, evecs, evecs_inverse, zero_ind, small_indices, diagonal, nu):
    diagonal[~small_indices] = 1 / (-evals[~small_indices] - 1j * nu)
    diagonal[zero_ind] = 0

    Fourier_G = evecs @ torch.diag(diagonal) @ evecs_inverse

    return Fourier_G


def benchmark_nu_sizes(nu_sizes):
    # Set up test system (2x2 Hamiltonian)
    H = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    c_ops = [torch.tensor([[0., 1.], [0., 0.]])]
    m_ops = {'0': torch.tensor([[0., 0.], [0., 1.]])}
    gamma = torch.tensor([1])
    beta = {'0': 1}

    # Generate Liouvillian and diagonalization
    L = liouvillian_superoperator(H, c_ops, m_ops, gamma, beta)
    evals, V, V_inv, zero_ind, small_mask, diag = diag_liouvillian(L)

    print(f"{'N_points':>10} | {'Loop_time (s)':>15} | {'Batch_time (s)':>15} | {'Speedup':>10}")
    print("-" * 60)

    for n_points in nu_sizes:
        nu_values = torch.linspace(-5.0, 5.0, n_points)

        # --- Loop timing ---
        t0 = time.perf_counter()
        G_loop = []
        for nu in nu_values:
            G = g_nu(evals, V, V_inv, zero_ind, small_mask, diag.clone(), nu)
            G_loop.append(G)
        G_loop = torch.stack(G_loop)
        t1 = time.perf_counter()
        loop_time = t1 - t0

        # --- Batch timing ---
        t2 = time.perf_counter()
        G_batch = g_nu_batch(evals, V, V_inv, zero_ind, small_mask, nu_values)
        t3 = time.perf_counter()
        batch_time = t3 - t2

        # --- Print results ---
        speedup = loop_time / batch_time if batch_time > 0 else float("inf")
        print(f"{n_points:>10} | {loop_time:>15.6f} | {batch_time:>15.6f} | {speedup:>10.2f}")

if __name__ == "__main__":
    nu_sizes = [2**n for n in range(7, 20)]
    benchmark_nu_sizes(nu_sizes)
