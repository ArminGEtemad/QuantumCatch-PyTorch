import torch
import time
import tracemalloc
from multichqc.spectra_calculator import liouvillian_superoperator, diag_liouvillian, g_nu_batch, g_nu_batch_unique

def g_nu(evals, evecs, evecs_inverse, zero_ind, small_indices, diagonal, nu):
    diagonal[~small_indices] = 1 / (-evals[~small_indices] - 1j * nu)
    diagonal[zero_ind] = 0

    Fourier_G = evecs @ torch.diag(diagonal) @ evecs_inverse

    return Fourier_G


def benchmark_nu_sizes(nu_sizes, device):
    # Set up test system (2x2 Hamiltonian)
    H = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    c_ops = [torch.tensor([[0., 1.], [0., 0.]])]
    m_ops = {'0': torch.tensor([[0., 0.], [0., 1.]])}
    gamma = torch.tensor([1])
    beta = {'0': 1}

    # Generate Liouvillian and diagonalization
    L = liouvillian_superoperator(H, c_ops, m_ops, gamma, beta, device)
    evals, V, V_inv, zero_ind, small_mask, diag = diag_liouvillian(L)

    print(f"{'N_points':>10} | {'Loop_time (s)':>15} | {'Batch_time (s)':>15} | {'Speedup':>10}")
    print("-" * 60)

    for n_points in nu_sizes:
        nu_values = torch.linspace(-5.0, 5.0, n_points).to(device)

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

import psutil
import os

def get_memory_mb():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 2)  # Resident Set Size in MB

def benchmark_unique(nu_sizes, device='cpu'):
    # Set up test system (2x2 Hamiltonian)
    H = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device)
    c_ops = [torch.tensor([[0., 1.], [0., 0.]], device=device)]
    m_ops = {'0': torch.tensor([[0., 0.], [0., 1.]], device=device)}
    gamma = torch.tensor([1], device=device)
    beta = {'0': 1}

    # Generate Liouvillian and diagonalization
    L = liouvillian_superoperator(H, c_ops, m_ops, gamma, beta, device)
    evals, V, V_inv, zero_ind, small_mask, diag = diag_liouvillian(L)

    print(f"{'N_points':>10} | {'batch_time (s)':>15} | {'unique_time (s)':>15} | {'Speedup':>10} | {'batch_mem (MB)':>15} | {'unique_mem (MB)':>15} | {'MemRatio':>10}")
    print("-" * 120)

    for n_points in nu_sizes:
        nu_values = torch.linspace(-2.0, 2.0, n_points, device=device)

        # --- Batch Timing & Memory ---
        torch.cuda.empty_cache() if device == 'cuda' else None
        mem_before = get_memory_mb()
        t0 = time.perf_counter()
        G_batch = g_nu_batch(evals, V, V_inv, zero_ind, small_mask, nu_values)
        t1 = time.perf_counter()
        mem_after = get_memory_mb()
        batch_time = t1 - t0
        batch_mem_mb = max(mem_after - mem_before, 0)

        # --- Unique Timing & Memory ---
        torch.cuda.empty_cache() if device == 'cuda' else None
        mem_before = get_memory_mb()
        t2 = time.perf_counter()
        G_unique = g_nu_batch_unique(evals, V, V_inv, zero_ind, small_mask, nu_values)
        t3 = time.perf_counter()
        mem_after = get_memory_mb()
        unique_time = t3 - t2
        unique_mem_mb = max(mem_after - mem_before, 0)

        # --- Print results ---
        speedup =  batch_time / unique_time if batch_time > 0 else float('inf')
        mem_ratio = batch_mem_mb / unique_mem_mb if unique_mem_mb > 0 else float('inf')
        print(f"{n_points:>10} | {batch_time:>15.6f} | {unique_time:>15.6f} | {speedup:>10.2f} | {batch_mem_mb:>15.2f} | {unique_mem_mb:>15.2f} | {mem_ratio:>10.2f}")


if __name__ == "__main__":
    device = 'cpu'
    nu_sizes = [2**n for n in range(7, 20)]
    which_benchmark = '2'

    if which_benchmark == '1':
        benchmark_nu_sizes(nu_sizes, device)
    if which_benchmark == '2':
        benchmark_unique(nu_sizes, device)

