# A minimal implementation of the PV-OSIM algorithm

import casadi as cs

def PvOsim(model, q, constraints):

    model.Omega = {}

     # First forward sweep
    for i in range(0, model.n_joints+1):

        model.Hi_A[i] = model.Hi[i]
        model.Ki_A[i] = cs.SX()
        model.Li_A[i] = cs.SX()
        model.constraints_supported[i] = []

    # Go through all the constraints to get the Kis
    for con in constraints:
        parent = con.parent
        model.Ki_A[parent] = cs.vertcat(model.Ki_A[parent], con.K)
        par_m = model.Li_A[parent].shape[0]
        m_i = con.K.shape[0]
        model.Li_A[parent] = cs.vertcat(cs.horzcat(model.Li_A[parent], cs.SX(par_m, m_i)), cs.SX(m_i, par_m + m_i))
        model.constraints_supported[parent].append(con.constraint_index) 

    
    for i in range(model.n_joints, 0, -1):
        
        Si = model.Si[i]
        model.Di[i] = Si.T @ model.Hi_A[i] @ Si
        model.Di_inv[i] = cs.inv(model.Di[i])
        
        model.Omega[i] = Si @ model.Di_inv[i] @ Si.T
        parent = model.parents[i]
        model.constraints_supported[parent] = model.constraints_supported[parent] + model.constraints_supported[i]
        if parent > 0:
            model.Pi[i] = cs.SX.eye(6) - model.Hi_A[i] @ model.Omega[i]
            model.Hi_A[parent] += model.Pi[i] @ model.Hi_A[i]

        if len(model.constraints_supported[i]) > 0:
            if parent > 0:
                model.Ki_A[parent] = cs.vertcat(model.Ki_A[parent], model.Ki_A[i] @ model.Pi[i].T)

            par_m = model.Li_A[parent].shape[0]
            m_i = model.Li_A[i].shape[0]
            model.Li_A[parent] = cs.vertcat(cs.horzcat(model.Li_A[parent], cs.SX(par_m, m_i)), cs.horzcat(cs.SX(m_i, par_m), model.Li_A[i] + model.Ki_A[i] @ model.Omega[i] @ model.Ki_A[i].T))

    return model.Li_A[0]