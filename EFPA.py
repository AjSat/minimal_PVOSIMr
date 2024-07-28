# A minimal implementation of the EFPA algorithm

import casadi as cs

def EFPA(model, q, constraints):

    K = {}
    OSIM = {}
    Omega = {}

    Omega_i_KjT = {}

    for i in range(0, model.n_joints+1):
        model.Hi_A[i] = model.Hi[i]
        model.constraints_supported[i] = []
    
    for con in constraints:
        parent = con.parent
        K[parent, con.constraint_index] = con.K
        model.constraints_supported[parent].append(con.constraint_index)

    for i in range(model.n_joints, 0, -1):
        
        parent = model.parents[i]
        model.constraints_supported[parent] += model.constraints_supported[i]
        Si = model.Si[i]

        model.Di[i] = Si.T @ model.Hi_A[i] @ Si
        model.Di_inv[i] = cs.inv(model.Di[i])
        Omega[parent, i] = Si @ model.Di_inv[i] @ Si.T

        if parent > 0:
            model.Pi[i] = cs.SX.eye(6) -  model.Hi_A[i] @ Si @ model.Di_inv[i] @ Si.T
            model.Hi_A[parent] += model.Pi[i]@model.Hi_A[i]

            for j in model.constraints_supported[i]:
                K[parent, j] = K[i, j]@model.Pi[i].T
    
    for i in range(1, model.n_joints+1):
        parent = model.parents[i]
        for j in model.constraints_supported[i]:
            # actual_con_ind = con_ind - model.n_joints
            if parent != 0:
                Omega_i_KjT[i, j] = model.Pi[i].T@Omega_i_KjT[parent, j] + Omega[parent, i]@K[i, j].T
                # Omega[0,i] = Omega[parent, i] + model.Pi[i].T@Omega[0, parent]@model.Pi[i]
            else:
                Omega_i_KjT[i, j] = Omega[0,i]@K[i,j].T

    for i in range(0, len(model.constraints_supported[0])):
        for j in range(i, len(model.constraints_supported[0])):
            i_link = constraints[i].parent
            j_link = constraints[j].parent
            i_ee = constraints[i].constraint_index
            j_ee = constraints[j].constraint_index
            cca_ij = getCCA(model, i_link, j_link)
            OSIM[i,j] = K[cca_ij, i_ee] @ Omega_i_KjT[cca_ij, j_ee]
                
    model.Omega = Omega
    model.K = K
    return OSIM


def getCCA(model, i, j):

    parents = model.parents
    
    i_anc = i
    j_anc = j

    while i_anc != j_anc:
        if i_anc > j_anc:
            i_anc = parents[i_anc]
        else:
            j_anc = parents[j_anc]

    return i_anc