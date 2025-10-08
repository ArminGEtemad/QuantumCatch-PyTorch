import torch

# Helper function to make omega out of nu
def freq_to_circfreq(freq):
    return freq * 2 * torch.pi

# Helper function
# commutation matrix to bring the liouvillian in the order we need
def comm_mat(n):
    a_fortran = torch.arange(n * n).reshape(n, n).t()
    w = a_fortran.reshape(-1)
    k_matrix = torch.eye(n * n)[w, :]
    return k_matrix

def liouvillian_superoperator(hamiltonian, cOps_list, measureOps_list, gamma_list, beta_list):
    COMPLEX64_TYPE = torch.cfloat

    # Using the shape of hamiltonian to calculate the shape of the Liovillian
    shape_hamiltonian = hamiltonian.shape[0]
    shape_liouvillian = shape_hamiltonian * shape_hamiltonian

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
