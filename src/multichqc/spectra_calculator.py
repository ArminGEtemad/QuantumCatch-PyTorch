# Copyright 2025 Armin Ghorbanietemad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
import torch
from typing import Tuple, Dict, Sequence, Optional
from torch import Tensor
import matplotlib.pyplot as plt
from itertools import permutations
from multichqc.input_config import SpectrumConfig
from multichqc.system_config import SystemConfig

COMPLEX64_TYPE = torch.cfloat

# Helper function to make omega out of nu
def freq_to_circfreq(freq: Tensor, backend:str) -> Tensor:
    omega = freq * 2 * torch.pi 
    return omega.to(backend)

# Helper function
# commutation matrix to bring the liouvillian in the order we need
def comm_mat(n: int) -> Tensor:
    a_fortran = torch.arange(n * n).reshape(n, n).t()
    w = a_fortran.reshape(-1)
    k_matrix = torch.eye(n * n)[w, :]
    return k_matrix


def liouvillian_superoperator(
    hamiltonian: Tensor,
    cOps_list: Sequence[Tensor],
    measureOps_dict: Dict[str, Tensor],
    gamma_list: Tensor,
    beta_dict: Dict[str, float],
    backend: str,
) -> Tensor:
    
    # Using the shape of hamiltonian to calculate the shape of the Liovillian
    shape_hamiltonian = hamiltonian.shape[0]
    shape_liouvillian = shape_hamiltonian * shape_hamiltonian
    measureOps_list = list(measureOps_dict.values())
    beta_list = list(beta_dict.values())

    # need commutation matrix to bring the order of the matrixized Liouvillian elements
    # to our conventional way
    commutation_matrix = comm_mat(shape_hamiltonian).to(COMPLEX64_TYPE)

    # matrixizing 
    # 1. identity matrix
    id_mat = torch.eye(shape_hamiltonian, dtype=COMPLEX64_TYPE)
    
    # 2. hamiltonian transposed
    hamiltonian = hamiltonian.to(COMPLEX64_TYPE)
    hamiltonian_t = hamiltonian.t().contiguous()
    
    # 3. collapse and measurement operators
    cOps_list_star = [torch.conj(cOps) for cOps in cOps_list]
    measureOps_list_star = [torch.conj(measureOps) for measureOps in measureOps_list]
    
    # 4. number of used operators
    size_cOps = len(cOps_list)
    size_measureOps = len(measureOps_list)

    # 5. actual calculation

    # ------------
    l1 = 1j * commutation_matrix @ (torch.kron(hamiltonian_t, id_mat) - torch.kron(id_mat, hamiltonian)) @ commutation_matrix

    l2_1 = torch.zeros((shape_liouvillian, shape_liouvillian), dtype=COMPLEX64_TYPE)
    for k in range(size_cOps):
        l2_1 += gamma_list[k] * torch.kron(cOps_list_star[k], cOps_list[k])
    l2_2 = torch.zeros((shape_liouvillian,shape_liouvillian), dtype=COMPLEX64_TYPE)
    for k in range(size_measureOps):
        l2_2 += beta_list[k]*beta_list[k] * torch.kron(measureOps_list_star[k], measureOps_list[k])
    l2 = commutation_matrix @ (l2_1+l2_2) @ commutation_matrix

    l3_1 = torch.zeros((shape_liouvillian,shape_liouvillian), dtype=COMPLEX64_TYPE)
    for k in range(size_cOps):
        l3_1 += gamma_list[k] / 2 * (torch.kron(id_mat, cOps_list_star[k].t().contiguous() @ cOps_list[k]) 
                                 + torch.kron(cOps_list[k].t().contiguous() @ cOps_list_star[k], id_mat))
    l3_2 = torch.zeros((shape_liouvillian,shape_liouvillian), dtype=COMPLEX64_TYPE)
    for k in range(size_measureOps):
        l3_2 += beta_list[k]*beta_list[k] / 2 * (torch.kron(id_mat, measureOps_list_star[k].t().contiguous() @ measureOps_list[k]) 
                                 + torch.kron(measureOps_list[k].t().contiguous() @ measureOps_list_star[k], id_mat))
    l3 = -(commutation_matrix @ (l3_1 + l3_2) @ commutation_matrix)

    liouvillian = l1 + l2 + l3
    # ------------

    return liouvillian.to(backend)

def rho_steady(
        hamiltonian: Tensor, 
        liouvillian: Tensor,
        backend: str,
) -> Tensor:

    # All the eigenvalues of the Liovillain are negative but the steady state must be the closes to zero
    shape_hamiltonian = hamiltonian.shape[0]
    evals, evecs = torch.linalg.eig(liouvillian.to(COMPLEX64_TYPE))
    zero_ind = torch.argmax(torch.real(evals))
    rho_steady = evecs[:, zero_ind]
    rho_steady = rho_steady / torch.trace(rho_steady.reshape((shape_hamiltonian, shape_hamiltonian)))
    rho_steady = torch.reshape(rho_steady, (shape_hamiltonian*shape_hamiltonian, 1))

    return rho_steady.to(backend)

def mathcal_a(
        hamiltonian: Tensor, 
        measureOps_dict: Dict[str, Tensor],
        backend:str,
) -> Dict[str, Tensor]:

    shape_hamiltonian = hamiltonian.shape[0]
    commutation_matrix = comm_mat(shape_hamiltonian).to(COMPLEX64_TYPE)
    id_mat = torch.eye(shape_hamiltonian).to(COMPLEX64_TYPE)

    a_dict = {}
    for key, measureOp in measureOps_dict.items():
        measureOp_star = torch.conj(measureOp)
        term = (1 / 2) * (torch.kron(id_mat, measureOp) + torch.kron(measureOp_star, id_mat))
        a_term = commutation_matrix @ term @ commutation_matrix

        a_dict[key] = a_term.to(backend)
        
    return a_dict

def mathcal_a_prim(
        hamiltonian: Tensor, 
        rho0: Tensor,
        measureOps_dict: Dict[str, Tensor],
        backend: str,
) -> Dict[str, Tensor]:
    
    # calculate the measurement superoperator
    mathcalA_dict = mathcal_a(hamiltonian, measureOps_dict, backend)

    shape_hamiltonian = hamiltonian.shape[0]
    shape_a_op = next(iter(mathcalA_dict.values())).shape[0]

    id_mat = torch.eye(shape_a_op).to(COMPLEX64_TYPE).to(backend)
    
    a_prim_dict = {}
    for key, mathcalA in mathcalA_dict.items():
        mathcalA_rho0 =  mathcalA @ rho0
        trace_mat = torch.trace(mathcalA_rho0.reshape(shape_hamiltonian, shape_hamiltonian)) * id_mat
        a_prim_term = mathcalA - trace_mat

        a_prim_dict[key] = a_prim_term
    return a_prim_dict

def diag_liouvillian(
        liouvillian: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: # lives wherever Liouvillian lives
    
    liouvillian = liouvillian.to(COMPLEX64_TYPE)
    evals, evecs = torch.linalg.eig(liouvillian)
    evecs_inverse = torch.linalg.inv(evecs)
    zero_ind = torch.argmax(torch.real(evals))
    small_indices = torch.abs(evals) < 1e-12
    diagonal = torch.zeros_like(evals)

    return evals, evecs, evecs_inverse, zero_ind, small_indices, diagonal

def g_nu_batch(
        evals: Tensor, 
        evecs: Tensor, 
        evecs_inv: Tensor, 
        zero_ind: Tensor, 
        small_mask: Tensor, 
        omega: Tensor
) -> Tensor: # this lives wherever Liouvillian lives
    
    B = omega.numel() # batching size

    denom = -evals.unsqueeze(0) - 1j * omega.unsqueeze(1)
    denom = denom.to(evals.dtype)

    inv = torch.zeros_like(denom)
    if (~small_mask).any():
        inv[:, ~small_mask] = 1.0 / denom[:, ~small_mask]

    inv[:, zero_ind] = 0.0

    evecsb = evecs.unsqueeze(0).expand(B, -1, -1) # batched 
    evecs_invb = evecs_inv.unsqueeze(0).expand(B, -1, -1) # batched

    tmp = evecsb * inv.unsqueeze(1)

    return torch.bmm(tmp, evecs_invb)

def s1(
    hamiltonian: Tensor, 
    measure_dict: Dict[str, Tensor], 
    beta_dict: Dict[str, float], 
    measureOrder_dict: Dict[str, list[Tuple[str]]],
    rho_0: Tensor,
    backend: str,
    key: str = 's1'
) -> Dict[str, Tensor]:
    shape_hamiltonian = hamiltonian.shape[0]

    # Normalize input to always be a list of tuples
    order_list = [
        (op,) if isinstance(op, str) else op
        for op in measureOrder_dict[key]
    ]

    used_keys = set(k for pair in order_list for k in pair)
    used_measure_dict = {k: measure_dict[k] for k in used_keys}

    a_dict = mathcal_a(hamiltonian, used_measure_dict, backend)

    result_dict = {}
    for op0_key in order_list:
        op_name = op0_key[0]
        a = a_dict[op_name]
        pref = beta_dict[op_name] ** 2

        a_rho = a @ rho_0
        tr = torch.trace(a_rho.reshape((shape_hamiltonian, shape_hamiltonian)))
        result = pref * tr

        if backend != 'cpu':
            result = result.cpu().numpy()

        result_dict[op_name] = result

    return result_dict


        
def s2_batched(
        hamiltonian: Tensor, 
        liouvillian: Tensor, 
        measure_dict: Dict[str, Tensor], 
        beta_dict: Dict[str, float], 
        measureOrder_dict: Dict[str, list[Tuple[str, str]]], 
        rho_0: Tensor,
        omega: Tensor,
        backend: str,
        key='s2',

) -> Dict[Tuple[str, str], Tensor]:
            
    # Get only the operators we’ll actually use
    used_keys = set(k for pair in measureOrder_dict[key] for k in pair)
    used_measure_dict = {k: measure_dict[k] for k in used_keys}
    
    # Compute all a' terms just once
    a_prim_dict = mathcal_a_prim(hamiltonian, rho_0, used_measure_dict, backend)

    # Diagonalize Liouvillian
    evals, evecs, evecs_inverse, zero_ind, small_indices, _ = diag_liouvillian(liouvillian)
    g_plus  = g_nu_batch(evals, evecs, evecs_inverse, zero_ind, small_indices,  omega)
    g_minus = g_nu_batch(evals, evecs, evecs_inverse, zero_ind, small_indices, -omega)

    shape_hamiltonian = hamiltonian.shape[0]
    b = omega.numel()

    result_dict = {}
    for op0_key, op1_key in measureOrder_dict[key]:
        a0 = a_prim_dict[op0_key]
        a1 = a_prim_dict[op1_key]

        a0_rho = a0 @ rho_0
        a1_rho = a1 @ rho_0

        a0_b = a0.unsqueeze(0).expand(b, -1, -1)
        a1_b = a1.unsqueeze(0).expand(b, -1, -1)
        a0_rho_b = a0_rho.unsqueeze(0).expand(b, -1, -1)
        a1_rho_b = a1_rho.unsqueeze(0).expand(b, -1, -1)

        s2_1 = torch.bmm(a0_b, torch.bmm(g_plus, a1_rho_b)).view(b, shape_hamiltonian, shape_hamiltonian)
        s2_2 = torch.bmm(a1_b, torch.bmm(g_minus, a0_rho_b)).view(b, shape_hamiltonian, shape_hamiltonian)

        tr_sum = torch.einsum('bii->b', s2_1) + torch.einsum('bii->b', s2_2)

        pref = (beta_dict[op0_key])**2 * (beta_dict[op1_key])**2
        if backend != 'cpu':
            result_dict[(op0_key, op1_key)] = (pref * tr_sum).cpu().numpy() # save the result back in CPU
        else:
            result_dict[(op0_key, op1_key)] = (pref * tr_sum)

    return result_dict

def s3_batched(
    hamiltonian: Tensor, 
    liouvillian: Tensor, 
    measure_dict: Dict[str, Tensor], 
    beta_dict: Dict[str, float], 
    measureOrder_dict: Dict[str, list[Tuple[str, str]]], 
    rho_0: Tensor,
    omega: Tensor,
    backend: str,
    key='s3',
    chunk_size: int | None = None,
) -> Dict[Tuple[str, str], Tensor]:
    
    # getting the keys that are actually used
    used_keys = set(k for pair in measureOrder_dict[key] for k in pair)
    used_measure_dict = {k: measure_dict[k] for k in used_keys}

    omega_1, omega_2 = torch.meshgrid(omega, omega, indexing='ij')
    end_shape = omega.numel()
    omega_1_f, omega_2_f = omega_1.flatten(), omega_2.flatten() 
    omega_3_f = -(omega_1_f + omega_2_f)

    omega_vec_full = [omega_1_f, omega_2_f, omega_3_f]

    b = omega_1_f.numel() # batching size
    if not chunk_size or chunk_size <= 0:
        chunk_size = b

    shape_hamiltonian = hamiltonian.shape[0]

    a_prim_dict = mathcal_a_prim(hamiltonian, rho_0, used_measure_dict, backend)

    evals, evecs, evecs_inverse, zero_ind, small_indices, _ = diag_liouvillian(liouvillian)
    # TODO: the needed permutations (REFACTOR LATER)
    perms = ((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0))

    result_dict = {}
    for op0_key, op1_key, op2_key in measureOrder_dict[key]:
        a0 = a_prim_dict[op0_key]
        a1 = a_prim_dict[op1_key]
        a2 = a_prim_dict[op2_key]

        a0_rho = a0 @ rho_0
        a1_rho = a1 @ rho_0
        a2_rho = a2 @ rho_0

        a_rho = [a0_rho, a1_rho, a2_rho]
        a_b = [a.unsqueeze(0) for a in (a0, a1, a2)]
        a_rho_b = [r.unsqueeze(0) for r in a_rho]

        res_flat = torch.zeros(b, dtype=COMPLEX64_TYPE, device=backend)

        for start in range(0, b, chunk_size):
            stop = min(start + chunk_size, b) 
            sl = slice(start, stop)
            b_b = stop - start 

            omega_block = [omega[sl] for omega in omega_vec_full] 

            G1 = [g_nu_batch(evals, evecs, evecs_inverse, zero_ind, small_indices, omega_block[n])
                for n in range(3)]
            
            G2 = [[None]*3 for _ in range(3)]
            for m in range(3):
                for n in range(3):
                    omega_mn = omega_block[m] + omega_block[n]
                    G2[m][n] = g_nu_batch(
                        evals, evecs, evecs_inverse, zero_ind, small_indices, omega_mn)

            chunk_out = torch.zeros(b_b, dtype=COMPLEX64_TYPE, device=backend)

            for l, m, n in perms:
                tmp = torch.bmm(G2[m][n], a_rho_b[l].expand(b_b, -1, -1))
                tmp = torch.bmm(a_b[m].expand(b_b, -1, -1), tmp)
                tmp = torch.bmm(G1[n], tmp)
                tmp = torch.bmm(a_b[n].expand(b_b, -1, -1), tmp).squeeze(-1)

                chunk_out += tmp.view(b_b, shape_hamiltonian, shape_hamiltonian)\
                                .diagonal(dim1=-2, dim2=-1).sum(-1)

            res_flat[sl] = chunk_out 
        pref = (beta_dict[op0_key])**2 * (beta_dict[op1_key])**2 * (beta_dict[op2_key])**2
        res_flat *= pref
        res_matrix = res_flat.view(end_shape, end_shape)
        res_matrix = res_matrix.t()

        # sending the results back to cpu if they are on gpu
        if backend != 'cpu':
            result_dict[(op0_key, op1_key, op2_key)] = res_matrix.cpu().numpy()
        else:
            result_dict[(op0_key, op1_key, op2_key)] = res_matrix
    return result_dict

# ----- Preparing s_4 calculations -----
def _prepare_freq_grid(freq : Tensor, backend: str) -> Tuple[Tensor, Tensor]:
    freq_1, freq_2 = torch.meshgrid(freq, freq, indexing='ij')
    return freq_to_circfreq(freq_1, backend), freq_to_circfreq(freq_2, backend)

def s4_batched(
    hamiltonian: Tensor, 
    liouvillian: Tensor, 
    measure_dict: Dict[str, Tensor], 
    beta_dict: Dict[str, float], 
    measureOrder_dict: Dict[str, list[Tuple[str, str]]], 
    rho_0: Tensor,
    freq: Tensor,
    backend: str,
    key='s4',
    chunk_size: int | None = None,
) -> Dict[Tuple[str, str, str, str], Tensor]:
    # 1. Prepare relevant operators
    used_keys = set(k for pair in measureOrder_dict[key] for k in pair)
    used_measure_dict = {k: measure_dict[k] for k in used_keys}

    # 2. Prepare frequency grid
    omega_1, omega_2 = _prepare_freq_grid(freq, backend)  # (F, F)
    end_shape, b = omega_1.shape[0], omega_1.numel()
    omega_1_f, omega_2_f = omega_1.flatten(), omega_2.flatten()
    omega_vec_full = torch.stack([omega_1_f, -omega_1_f, omega_2_f, -omega_2_f])  # (4, B)

    # 3. Liouville preliminaries
    d = hamiltonian.shape[0]
    a_prim_dict = mathcal_a_prim(hamiltonian, rho_0, used_measure_dict, backend)
    evals, evecs, evecs_inverse, zero_ind, small_indices, _ = diag_liouvillian(liouvillian)
    perms = list(permutations(range(4), 4))

    # 4. Results container
    result_dict = {}

    # 5. Iterate over all operator combinations
    for op0_key, op1_key, op2_key, op3_key in measureOrder_dict[key]:
        a0 = a_prim_dict[op0_key]
        a1 = a_prim_dict[op1_key]
        a2 = a_prim_dict[op2_key]
        a3 = a_prim_dict[op3_key]

        a0_rho = a0 @ rho_0
        a1_rho = a1 @ rho_0
        a2_rho = a2 @ rho_0
        a3_rho = a3 @ rho_0
        a_rho = [a0_rho, a1_rho, a2_rho, a3_rho]

        a_b = [a.unsqueeze(0) for a in (a0, a1, a2, a3)]
        a_rho_b = [r.unsqueeze(0) for r in a_rho]

        res_flat = torch.zeros(b, dtype=COMPLEX64_TYPE, device=backend)

        # Normalize chunk size
        if not chunk_size or chunk_size <= 0:
            chunk_size = b

        # Inner compute block
        def _do_block(idx, stop):
            sl = slice(idx, stop)
            w = omega_vec_full[:, sl]
            b_b = stop - idx

            # Single-index Green's
            G1 = [g_nu_batch(evals, evecs, evecs_inverse, zero_ind, small_indices, w[j]) for j in range(4)]

            # Pair and triple caches
            G2, G3 = {}, {}
            out = torch.zeros(b_b, dtype=COMPLEX64_TYPE, device=backend)

            for k, l, m, n in perms:
                key2 = (m, n)
                if key2 not in G2:
                    G2[key2] = g_nu_batch(evals, evecs, evecs_inverse, zero_ind, small_indices, w[m] + w[n])
                key3 = (l, m, n)
                if key3 not in G3:
                    G3[key3] = g_nu_batch(evals, evecs, evecs_inverse, zero_ind, small_indices, w[l] + w[m] + w[n])

                tmp = torch.bmm(G3[key3], a_rho_b[k].expand(b_b, -1, -1))
                tmp = torch.bmm(a_b[l].expand(b_b, -1, -1), tmp)
                tmp = torch.bmm(G2[key2], tmp)
                tmp = torch.bmm(a_b[m].expand(b_b, -1, -1), tmp)
                tmp = torch.bmm(G1[n], tmp)
                tmp = torch.bmm(a_b[n].expand(b_b, -1, -1), tmp).squeeze(-1)

                out += tmp.view(b_b, d, d).diagonal(dim1=-2, dim2=-1).sum(-1)

            pref = (
                beta_dict[op0_key] ** 2 *
                beta_dict[op1_key] ** 2 *
                beta_dict[op2_key] ** 2 *
                beta_dict[op3_key] ** 2
            )

            return out * pref

        # Chunked processing
        if chunk_size and chunk_size < b:
            for b0 in range(0, b, chunk_size):
                stop = min(b0 + chunk_size, b)
                res_flat[b0:stop] = _do_block(b0, stop)
        else:
            res_flat = _do_block(0, b)

        # Store result (with all 4 keys)
        res_view = res_flat.view(end_shape, end_shape)
        if backend != 'cpu':
            res_view = res_view.cpu().numpy()
        result_dict[(op0_key, op1_key, op2_key, op3_key)] = res_view

    return result_dict

def matrices_S(liouvillian: Tensor, zero_ind: int) -> Tensor:

    dim = liouvillian.shape[0]
    dtype = liouvillian.dtype
    dev = liouvillian.device

    S = torch.zeros((dim, dim, dim), dtype=dtype, device=dev)
    idx = torch.arange(dim, device=dev)
    mask = idx != zero_ind
    S[mask, idx[mask], idx[mask]] = 1.0
    return S


def _small_s_for_ops(
    hamiltonian: Tensor,
    liouvillian: Tensor,
    aprim_list: list[Tensor],   # ordered [a0', a1', a2', a3'] for THIS operator quad
    rho0: Tensor,
    evecs: Tensor,
    evecs_inv: Tensor,
    zero_ind: int,
) -> tuple[Tensor, list[tuple[int,int]]]:

    dev = liouvillian.device
    dtype = liouvillian.dtype
    dim = liouvillian.shape[0]
    hdim = hamiltonian.shape[0]

    # all pairs with n ≠ k on 4 operators
    nk_pairs = [(n, k) for n in range(4) for k in range(4) if n != k]

    # projectors in Liouvillian eigenbasis (skip steady-state)
    S = matrices_S(liouvillian, zero_ind)

    D = torch.stack([evecs @ S_alpha @ evecs_inv for S_alpha in S])

    # pre-multiply ρ0 once
    aprim_rho0 = [A @ rho0 for A in aprim_list]

    # s has one row per Liouvillian eigenmode, zeroing steady state row
    s = torch.zeros((dim, len(nk_pairs)), dtype=dtype, device=dev)
    for alpha in range(dim):
        if alpha == zero_ind:
            continue
        D_alpha = D[alpha]
        for p_idx, (n, k) in enumerate(nk_pairs):
            term = aprim_list[n] @ D_alpha @ aprim_rho0[k]
            s[alpha, p_idx] = term.reshape(hdim, hdim).trace()

    return s, nk_pairs

def s4_corr_batched(
    hamiltonian: Tensor, 
    liouvillian: Tensor, 
    rho_0: Tensor,
    measure_dict: Dict[str, Tensor], 
    beta_dict: Dict[str, float], 
    measureOrder_dict: Dict[str, list[Tuple[str, str, str, str]]],  
    freq: Tensor,
    backend: str,
    chunk_size: int | None = None,
    key: str = 's4',
    which: str = "1",   # "1" => (lambda_a, lambda_a, lambda_a+lambda_b),
                        # "2" => (lambda_b, lambda_a, lambda_a+lambda_b)
) -> Dict[Tuple[str, str, str, str], Tensor]:

    # --- prep freq grid (F,F) → flattened vectors
    omega_1, omega_2 = _prepare_freq_grid(freq, backend)
    end_shape = omega_1.shape[0]
    omega_1_f, omega_2_f = omega_1.flatten(), omega_2.flatten()
    b = omega_1_f.numel()
    Omega = torch.stack([omega_1_f, -omega_1_f, omega_2_f, -omega_2_f])

   
    evals, evecs, evecs_inverse, zero_ind, small_indices, _ = diag_liouvillian(liouvillian)
    dev = liouvillian.device

    # mask out steady-state eigenvalue
    mask = torch.arange(evals.numel(), device=dev) != zero_ind
    lam  = evals[mask]
    d_L  = lam.numel()

    lam_a = lam.view(d_L, 1, 1)
    lam_b = lam.view(1, d_L, 1)

    if not chunk_size or chunk_size <= 0:
        chunk_size = b

    perms4 = list(permutations(range(4), 4))

    result_dict = {}

    # Build only the operators we’ll actually use
    used_keys = set(k for quad in measureOrder_dict[key] for k in quad)
    used_measure_dict = {k: measure_dict[k] for k in used_keys}

    a_prim_dict = mathcal_a_prim(hamiltonian, rho_0, used_measure_dict, backend)

    # for each operator quad, compute its correction
    for op0_key, op1_key, op2_key, op3_key in measureOrder_dict[key]:
        # order the four a' according to this tuple
        a0 = a_prim_dict[op0_key]
        a1 = a_prim_dict[op1_key]
        a2 = a_prim_dict[op2_key]
        a3 = a_prim_dict[op3_key]
        a_list = [a0, a1, a2, a3]

        # s_{α}^{n,k} table for this ordering
        s_nk_full, nk_pairs = _small_s_for_ops(
            hamiltonian, liouvillian, a_list, rho_0, evecs, evecs_inverse, zero_ind
        )
        # keep only non-steady rows
        s_nk = s_nk_full[mask]
        pair2idx = {p: i for i, p in enumerate(nk_pairs)}

        pref = (
            beta_dict[op0_key] ** 2 *
            beta_dict[op1_key] ** 2 *
            beta_dict[op2_key] ** 2 *
            beta_dict[op3_key] ** 2
        )

        # accumulator over the whole grid
        res_flat = torch.zeros(b, dtype=COMPLEX64_TYPE, device=dev)

        # process frequency blocks
        for start in range(0, b, chunk_size):
            stop = min(start + chunk_size, b)
            sl = slice(start, stop)
            b_b = stop - start

            nu = Omega[:, sl] 
            nu_i = [1j * nu[j].view(1, 1, b_b) for j in range(4)]

            block_sum = torch.zeros(b_b, dtype=COMPLEX64_TYPE, device=dev)

            for n, m, l, k in perms4:
                # pick the two s-columns by pair indices
                s_alpha = s_nk[:, pair2idx[(n, k)]]
                s_beta  = s_nk[:, pair2idx[(m, l)]]

                # outer product over α and β (d_L, d_L, 1)
                num = s_alpha.view(d_L, 1, 1) * s_beta.view(1, d_L, 1)

                # frequency combinations as per publication
                nu1 = nu_i[l] + nu_i[m] + nu_i[n]
                nu2 = nu_i[m] + nu_i[n]
                nu3 = nu_i[n]

                if which == "1":
                    denom = (lam_a + nu1) * (lam_a + nu3) * (lam_a + lam_b + nu2)
                else:  # which == "2"
                    denom = (lam_b + nu1) * (lam_a + nu3) * (lam_a + lam_b + nu2)

                block_sum += (num / denom).sum((0, 1))

            res_flat[sl] = block_sum

        res = pref * res_flat.view(end_shape, end_shape)

        if backend != 'cpu':
            res = res.cpu().numpy()
        result_dict[(op0_key, op1_key, op2_key, op3_key)] = res

    return result_dict

def _sum_spectra_dicts(*spectra_dicts: Dict[Tuple[str, str, str, str], object]
) -> Dict[Tuple[str, str, str, str], object]:

    if not spectra_dicts:
        return {}

    # enforce identical keysets
    keyset = set(spectra_dicts[0].keys())
    for i, d in enumerate(spectra_dicts[1:], start=2):
        if set(d.keys()) != keyset:
            missing_left = keyset - set(d.keys())
            missing_right = set(d.keys()) - keyset
            raise KeyError(
                f"Spectrum dict #{i} has mismatched keys.\n"
                f"Missing in dict #{i}: {sorted(missing_left)}\n"
                f"Extra in dict #{i}: {sorted(missing_right)}"
            )

    # sum values per key
    out: Dict[Tuple[str, str, str, str], object] = {}
    for k in keyset:
        acc = spectra_dicts[0][k]
        for d in spectra_dicts[1:]:
            acc = acc + d[k]
        out[k] = acc
    return out


def S4_total(
    hamiltonian: Tensor, 
    liouvillian: Tensor, 
    measure_dict: Dict[str, Tensor], 
    beta_dict: Dict[str, float], 
    measureOrder_dict: Dict[str, list[Tuple[str, str]]], 
    rho_0: Tensor,
    freq: Tensor,
    backend: str,
    key='s4',
    chunk_size: int | None = None,
) -> Dict[Tuple[str, str, str, str], object]:

    S_main = s4_batched(
        hamiltonian, liouvillian, measure_dict,
        beta_dict, measureOrder_dict, rho_0,
        freq, backend, key=key, chunk_size=chunk_size
    )

    S_c1 = s4_corr_batched(
        hamiltonian, liouvillian, rho_0,
        measure_dict, beta_dict, measureOrder_dict,
        freq, backend, chunk_size=chunk_size, key=key, which="1"
    )

    S_c2 = s4_corr_batched(
        hamiltonian, liouvillian, rho_0,
        measure_dict, beta_dict, measureOrder_dict,
        freq, backend, chunk_size=chunk_size, key=key, which="2"
    )

    sum_spec = _sum_spectra_dicts(S_main, S_c1, S_c2)
    out = {k: v.T for k, v in sum_spec.items()}
    return out



class SpectrumCalculator:
    def __init__(self, spe_config: SpectrumConfig, sys_config: SystemConfig):
        self.spe_config = spe_config
        self.sys_config = sys_config
        self.backend = self.spe_config.backend

        self.liouvillian = self._compute_liouvillian()
        self.freq = torch.linspace(spe_config.f_min, spe_config.f_max, spe_config.resolution)
        self.omega = freq_to_circfreq(self.freq, self.backend)
        self.rho_0 = rho_steady(sys_config.hamiltonian, self.liouvillian, self.backend)
        self.requested_order_list = list(spe_config.order_combi.keys())
        self.s1 = None
        self.s2 = None
        self.s3 = None
        self.s4 = None

        if 's1' in self.requested_order_list:
            self.s1 = self._compute_s1()
        if 's2' in self.requested_order_list:
            self.s2 = self._compute_s2()
        if 's3' in self.requested_order_list:
            self.s3 = self._compute_s3()
        if 's4' in self.requested_order_list:
            self.s4 = self._compute_s4()

    def _compute_liouvillian(self):
        return liouvillian_superoperator(
            self.sys_config.hamiltonian,
            self.sys_config.collapse_ops,
            self.sys_config.measurement_ops,
            self.sys_config.collapse_strength,
            self.sys_config.measurement_strength,
            self.backend
        )
    
    def _compute_s1(self):
        return s1(
            self.sys_config.hamiltonian, 
            self.sys_config.measurement_ops, 
            self.sys_config.measurement_strength,
            self.spe_config.order_combi,
            self.rho_0,
            self.backend,
            key='s1'
        )

    def _compute_s2(self):
        return s2_batched(
            self.sys_config.hamiltonian,
            self.liouvillian,
            self.sys_config.measurement_ops,
            self.sys_config.measurement_strength,
            self.spe_config.order_combi,
            self.rho_0,
            self.omega,
            self.backend,
            key='s2'
        )
    
    def _compute_s3(self):
        return s3_batched(
            self.sys_config.hamiltonian,
            self.liouvillian,
            self.sys_config.measurement_ops,
            self.sys_config.measurement_strength,
            self.spe_config.order_combi,
            self.rho_0,
            self.omega,
            self.backend,
            key='s3',
            chunk_size=self.spe_config.s3_block
        )
    
    def _compute_s4(self):
        return S4_total(
            self.sys_config.hamiltonian,
            self.liouvillian,
            self.sys_config.measurement_ops,
            self.sys_config.measurement_strength,
            self.spe_config.order_combi,
            self.rho_0,
            self.freq, # this is correct TODO: consistent with other functions
            self.backend,
            key='s4',
            chunk_size=self.spe_config.s4_block
        )
