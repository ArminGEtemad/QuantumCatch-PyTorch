from dataclasses import dataclass, field
import torch
from typing import Tuple, Dict, Sequence, Optional
from torch import Tensor
import matplotlib.pyplot as plt
from multichqc.input_config import SpectrumConfig
from multichqc.system_config import SystemConfig

COMPLEX64_TYPE = torch.cfloat

# Helper function to make omega out of nu
def freq_to_circfreq(freq: Tensor) -> Tensor:
    return freq * 2 * torch.pi

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

    return liouvillian

def rho_steady(
        hamiltonian: Tensor, 
        liouvillian: Tensor
) -> Tensor:

    # All the eigenvalues of the Liovillain are negative but the steady state must be the closes to zero
    shape_hamiltonian = hamiltonian.shape[0]
    evals, evecs = torch.linalg.eig(liouvillian.to(COMPLEX64_TYPE))
    zero_ind = torch.argmax(torch.real(evals))
    rho_steady = evecs[:, zero_ind]
    rho_steady = rho_steady / torch.trace(rho_steady.reshape((shape_hamiltonian, shape_hamiltonian)))
    rho_steady = torch.reshape(rho_steady, (shape_hamiltonian*shape_hamiltonian, 1))

    return rho_steady

def mathcal_a(
        hamiltonian: Tensor, 
        measureOps_dict: Dict[str, Tensor]
) -> Dict[str, Tensor]:

    shape_hamiltonian = hamiltonian.shape[0]
    commutation_matrix = comm_mat(shape_hamiltonian).to(COMPLEX64_TYPE)
    id_mat = torch.eye(shape_hamiltonian).to(COMPLEX64_TYPE)

    a_dict = {}
    for key, measureOp in measureOps_dict.items():
        measureOp_star = torch.conj(measureOp)
        term = (1 / 2) * (torch.kron(id_mat, measureOp) + torch.kron(measureOp_star, id_mat))
        a_term = commutation_matrix @ term @ commutation_matrix

        a_dict[key] = a_term
        
    return a_dict 

def mathcal_a_prim(
        hamiltonian: Tensor, 
        liouvillian: Tensor, 
        measureOps_dict: Dict[str, Tensor]
) -> Dict[str, Tensor]:
    
    # calculate the measurement superoperator
    mathcalA_dict = mathcal_a(hamiltonian, measureOps_dict)

    shape_hamiltonian = hamiltonian.shape[0]
    shape_a_op = next(iter(mathcalA_dict.values())).shape[0]

    id_mat = torch.eye(shape_a_op).to(COMPLEX64_TYPE)
    
    # calculate the steady state
    rho0 = rho_steady(hamiltonian, liouvillian)
    
    a_prim_dict = {}
    for key, mathcalA in mathcalA_dict.items():
        mathcalA_rho0 =  mathcalA @ rho0
        trace_mat = torch.trace(mathcalA_rho0.reshape(shape_hamiltonian, shape_hamiltonian)) * id_mat
        a_prim_term = mathcalA - trace_mat

        a_prim_dict[key] = a_prim_term
    return a_prim_dict

def diag_liouvillian(
        liouvillian: Tensor
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    
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
        nu: Tensor
) -> Tensor:
    
    B = nu.numel() # batching size

    denom = -evals.unsqueeze(0) - 1j * nu.unsqueeze(1)
    denom = denom.to(evals.dtype)

    inv = torch.zeros_like(denom)
    if (~small_mask).any():
        inv[:, ~small_mask] = 1.0 / denom[:, ~small_mask]

    inv[:, zero_ind] = 0.0

    evecsb = evecs.unsqueeze(0).expand(B, -1, -1) # batched 
    evecs_invb = evecs_inv.unsqueeze(0).expand(B, -1, -1) # batched

    tmp = evecsb * inv.unsqueeze(1)

    return torch.bmm(tmp, evecs_invb)

def g_nu_batch_unique(
        evals: Tensor,
        evecs: Tensor,
        evecs_inv: Tensor, 
        zero_ind: Tensor, 
        small_mask: Tensor, 
        nu: Tensor
) -> Tensor:

    nu_flat = nu.reshape(-1)
    nu_cpu  = nu_flat.detach().to(torch.float64).cpu().contiguous()
    uniq_vals, inverse = torch.unique(nu_cpu, return_inverse=True)

    if uniq_vals.numel() == 0:
        # Degenerate case; should not happen, but keep it safe
        m = evals.numel()
        return torch.zeros((0, m, m), dtype=evecs.dtype, device=evecs.device)

    uniq = uniq_vals.to(device=nu.device, dtype=nu.dtype)
    G_uniq = g_nu_batch(evals, evecs, evecs_inv, zero_ind, small_mask, uniq)

    # Reindex back to original order
    inverse_dev = inverse.to(G_uniq.device)
    out = G_uniq.index_select(0, inverse_dev)
    return out.view(nu_flat.shape[0], *G_uniq.shape[1:])

def s2_batched(
        hamiltonian: Tensor, 
        liouvillian: Tensor, 
        measure_dict: Dict[str, Tensor], 
        beta_dict: Dict[str, float], 
        measureOrder_dict: Dict[str, list[Tuple[str, str]]], 
        rho_0: Tensor,
        omega: Tensor,
        key='s2',
) -> Dict[Tuple[str, str], Tensor]:
            
    # Get only the operators we’ll actually use
    used_keys = set(k for pair in measureOrder_dict[key] for k in pair)
    used_measure_dict = {k: measure_dict[k] for k in used_keys}
    
    # Compute all a' terms just once
    a_prim_dict = mathcal_a_prim(hamiltonian, liouvillian, used_measure_dict)

    # Diagonalize Liouvillian
    evals, evecs, evecs_inverse, zero_ind, small_indices, _ = diag_liouvillian(liouvillian)
    g_plus  = g_nu_batch_unique(evals, evecs, evecs_inverse, zero_ind, small_indices,  omega)
    g_minus = g_nu_batch_unique(evals, evecs, evecs_inverse, zero_ind, small_indices, -omega)

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
        result_dict[(op0_key, op1_key)] = pref * tr_sum

    return result_dict

@dataclass
class SpectrumCalculator:
    spe_config: SpectrumConfig
    sys_config: SystemConfig

    liouvillian: Tensor = field(init=False)
    d_liouvillian: Tuple[Tensor, Tensor, Tensor, 
                         Tensor, Tensor, Tensor] = field(init=False)
    rho_0: Tensor = field(init=False)
    omega: Tensor = field(init=False)
    freq: Tensor = field(init=False)
    requested_order_list: list[str] = field(init=False)
    s2: Optional[Dict[Tuple[str, str], Tensor]] = field(init=False, default=None)

    def __post_init__(self):

        self.liouvillian = liouvillian_superoperator(
            self.sys_config.hamiltonian, 
            self.sys_config.collapse_ops, 
            self.sys_config.measurement_ops, 
            self.sys_config.collapse_strength, 
            self.sys_config.measurement_strength
        )

        self.freq = torch.linspace(
            self.spe_config.f_min, 
            self.spe_config.f_max, 
            self.spe_config.resolution
        )

        self.omega = freq_to_circfreq(self.freq)
        self.rho_0 = rho_steady(self.sys_config.hamiltonian, self.liouvillian)
        self.requested_order_list = list(self.spe_config.order_combi.keys())

        if 's2' in self.requested_order_list:
            self.s2 = s2_batched(
                self.sys_config.hamiltonian,
                self.liouvillian,
                self.sys_config.measurement_ops,
                self.sys_config.measurement_strength,
                self.spe_config.order_combi,
                self.rho_0,
                self.omega,
                key='s2'
            )