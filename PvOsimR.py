# A minimal implementation of the PvOsimR algorithm

import casadi as cs

def PvOsimR(model, q, constraints):

    K = {}
    Delassus = {}
    Omega = {}
    P = {}

    for i in range(0, model.n_joints+1):
        model.Hi_A[i] = model.Hi[i]
        model.constraints_supported[i] = []

    N = compute_PvOsimRMetaData(model, constraints)
    
    for con in constraints:
        parent = con.parent
        P[parent, con.constraint_index] = con.K.T
        m_i = con.K.shape[0]
        Omega[parent, con.constraint_index] = cs.SX(m_i, m_i)

    for i in range(model.n_joints, 0, -1):

        if i in N:
            P[i,i] = cs.SX.eye(6)
            Omega[i,i] = cs.SX(6,6)
        parent = model.parents[i]
        Si = model.Si[i]

        model.Di[i] = Si.T @ model.Hi_A[i] @ Si
        model.Di_inv[i] = cs.inv(model.Di[i])
        Omega[parent, i] = Si @ model.Di_inv[i] @ Si.T

        if parent > 0:
            P[parent,i] = cs.SX.eye(6) -  model.Hi_A[i] @ Omega[parent, i]
            model.Hi_A[parent] += P[parent,i]@model.Hi_A[i]

            if (len(model.constraints_supported[i]) > 0):
                P[parent, model.D[i]] = P[parent, i] @ P[i, model.D[i]]

        if len(model.constraints_supported[i]) > 0:
            Omega[parent, model.D[i]] = Omega[i, model.D[i]] + P[i, model.D[i]].T @ Omega[parent, i] @ P[i, model.D[i]]

            # for j in model.constraints_supported[i]:
    #             K[parent, j] = K[i, j]@model.Pi[i].T

    for ind in range(1, len(model.N)):
        i = model.N[ind]
        Ai = model.A[model.parents[i]]
        if Ai != 0:
            Omega[0,i] = Omega[Ai, i] + P[Ai, i].T @ Omega[0, Ai] @ P[Ai, i]

    for con in constraints:
        i = con.constraint_index
        Ai = model.A[con.parent]
        if Ai != 0:
            Omega[0, i] = Omega[Ai, i] + P[Ai, i].T @ Omega[0, Ai] @ P[Ai, i]
        

    for con in constraints:
        i = con.constraint_index
        j = con.parent
        Apar = model.A[j]
        if Apar != 0:
            K[Apar, i] = P[Apar, i].T
            j = Apar
            Apar = model.A[model.parents[j]]
            while Apar != 0:
                K[Apar, i] = K[j, i]@P[Apar, j].T
                j = Apar
                Apar = model.A[model.parents[j]]

    # Assembe the Delassus matrix
    for i in range(0, len(constraints)):
        i_link = constraints[i].constraint_index
        for j in range(i, len(constraints)):
            if i == j:
                Delassus[i,j] = Omega[0,i_link]
            else:
                j_link = constraints[j].constraint_index
                cca_ij = getCCA(model, constraints[i].parent, constraints[j].parent)
                Delassus[i,j] = K[cca_ij, i_link] @ Omega[0, cca_ij] @ K[cca_ij, j_link].T

    return Delassus

def compute_PvOsimRMetaData(model, constraints):
    
    Eps = []
    constraints_supported_dim = [0]*(model.n_joints + 1)
    A = [0]*(model.n_joints + 1 + len(constraints))
    D = [None]*(model.n_joints + 1)

    N = set()
    for con in constraints:
        Eps.append(con.constraint_index)
        parent = con.parent
        model.constraints_supported[parent].append(con.constraint_index)
        m_con = con.K.shape[0]
        if constraints_supported_dim[parent] > 0 and parent not in N:
            N.add(parent)

        D[parent] = con.constraint_index
        constraints_supported_dim[parent] += m_con

    for i in range(model.n_joints, 0, -1):
        parent = model.parents[i]
        
        if constraints_supported_dim[i] > 0 and constraints_supported_dim[parent] > 0 and parent not in N:
            N.add(parent)
        constraints_supported_dim[parent] += constraints_supported_dim[i]
        model.constraints_supported[parent] += model.constraints_supported[i]

        if constraints_supported_dim[i] > 0:
            if i in N:
                D[i] = i
            D[parent] = D[i]

    N.add(0)

    A[0] = 0
    for i in range(1, model.n_joints + 1):
        if i in N:
            A[i] = i
        else:
            A[i] = A[model.parents[i]]

    for con in constraints:
        parent = con.parent
        if A[parent] == parent:
            A[con.constraint_index] = parent
        else:
            A[con.constraint_index] = A[parent] 
        
    N_list = sorted(list(N))

    model.N = N_list
    model.D = D
    model.A = A

    return N
        

def getCCA(model, i, j):

    parents = model.parents
    
    i_anc = i
    j_anc = j

    while i_anc != j_anc:
        if i_anc > j_anc:
            i_anc = model.A[parents[i_anc]]
        else:
            j_anc = model.A[parents[j_anc]]

    return i_anc