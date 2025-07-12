# A minimal implementation of the PV-OSIM algorithm, generalized for multi-link constraints

import casadi as cs

def PvOsim(model, q, constraints):

    model.Omega = {}

     # First forward sweep
    for i in range(0, model.n_joints+1):

        model.Hi_A[i] = model.Hi[i]
        model.Ki_A[i] = {}
        model.constraints_supported[i] = []

    model.Li_A = {}
    # Go through all the constraints to get the Kis
    for con in constraints:
        constraint_id = con.constraint_index
        for link_idx, K_mat in con.links:
            model.Ki_A[link_idx][constraint_id] = K_mat
            
        
        m_i = K_mat.shape[0]
        model.Li_A[constraint_id,constraint_id] = cs.SX(m_i,m_i)
        for con_j in constraints:
            con_j_id = con_j.constraint_index
            m_j = con_j.links[0][1].shape[0]
            if con_j_id > constraint_id:
                model.Li_A[constraint_id, con_j_id] = cs.SX(m_i, m_j)
            elif con_j_id < constraint_id:
                model.Li_A[con_j_id, constraint_id] = cs.SX(m_j, m_i)

    # Backward sweep
    for i in range(model.n_joints, 0, -1):
        Si = model.Si[i]
        model.Di[i] = Si.T @ model.Hi_A[i] @ Si
        model.Di_inv[i] = cs.inv(model.Di[i])
        
        model.Omega[i] = Si @ model.Di_inv[i] @ Si.T
        parent = model.parents[i]
        
        if parent > 0:
            model.Pi[i] = cs.SX.eye(6) - model.Hi_A[i] @ model.Omega[i]
            model.Hi_A[parent] += model.Pi[i] @ model.Hi_A[i]

            # Propagate Ki_A
            for con_id in model.Ki_A[i]:
                if con_id in model.Ki_A[parent]:
                    model.Ki_A[parent][con_id] += model.Ki_A[i][con_id] @ model.Pi[i].T
                else:
                    model.Ki_A[parent][con_id] = model.Ki_A[i][con_id] @ model.Pi[i].T

        # Propagate and update Li_A
        for con_i_id in model.Ki_A[i]:
            model.Li_A[con_i_id, con_i_id] += model.Ki_A[i][con_i_id] @ model.Omega[i] @ model.Ki_A[i][con_i_id].T
            for con_j_id in model.Ki_A[i]:
                if con_j_id > con_i_id:
                    model.Li_A[con_i_id, con_j_id] += model.Ki_A[i][con_i_id] @ model.Omega[i] @ model.Ki_A[i][con_j_id].T
                elif con_j_id < con_i_id:
                    model.Li_A[con_j_id, con_i_id] += model.Ki_A[i][con_j_id] @ model.Omega[i] @ model.Ki_A[i][con_i_id].T

    # Assemble the final Delassus matrix
    return model.Li_A