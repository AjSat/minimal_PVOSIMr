# A minimal implementation of the PvOsimR algorithm, generalized for multi-link constraints

import casadi as cs
from collections import defaultdict

def get_ancestors(model, i):
    ancestors = []
    curr = i
    while curr != 0:
        curr = model.parents[curr]
        ancestors.append(curr)
    return ancestors

def get_coupling_pairs(constraints_from_descendants):
    coupling_pairs = {}
    for i in range(len(constraints_from_descendants)):
        for j in range(i+1, len(constraints_from_descendants)):
            for con1 in constraints_from_descendants[i]:
                for con2 in constraints_from_descendants[j]:
                    if con1 <= con2:
                        if con1 not in coupling_pairs:
                            coupling_pairs[con1] = []
                        if con2 not in coupling_pairs[con1]: coupling_pairs[con1].append(con2)
                    else:
                        if con2 not in coupling_pairs:
                            coupling_pairs[con2] = []
                        if con1 not in coupling_pairs[con2]: coupling_pairs[con2].append(con1)
    return coupling_pairs


def compute_metadata(model, constraints):
    constraints_on_link = defaultdict(set)
    constraints_from_descendants = defaultdict(list)
    mathcalD = {}
    mathcalN = set()
    constraint_ancestor = {} # stores the ancestor branching point for each constraint
    mathcalA = {}
    for i in range(model.n_joints + 1):
        mathcalD[i] = 0

    K_direct = defaultdict(dict)
    for con in constraints:
        for link_idx, K_mat in con.links:
            if link_idx not in constraints_from_descendants:
                constraints_from_descendants[link_idx] = []
            #     mathcalD[link_idx] = con.constraint_index
            # else:
            mathcalD[link_idx] = link_idx
            mathcalN.add(link_idx)
            constraints_from_descendants[link_idx].append(set([con.constraint_index]))
            constraints_on_link[link_idx].add(con.constraint_index)
            K_direct[link_idx][con.constraint_index] = K_mat

    for i in range(model.n_joints, 0, -1):
        if mathcalD[i] == 0:
            continue
        parent = model.parents[i]
        if mathcalD[parent] != 0:
            if mathcalD[parent] != parent:
                mathcalN.add(parent)
                mathcalA[mathcalD[parent]] = parent
                constraints_from_descendants[parent].append(constraints_on_link[mathcalD[parent]])

                # # Compute intersection and update constraint_ancestor
                # intersection = constraints_on_link[mathcalD[parent]] & constraints_on_link[mathcalD[i]]
                # for constraint in intersection:
                #     constraint_ancestor[constraint, mathcalD[parent]] = parent
                
                constraints_on_link[parent].update(constraints_on_link[mathcalD[parent]])
                mathcalD[parent] = parent
            mathcalA[mathcalD[i]] = parent
            constraints_from_descendants[parent].append(constraints_on_link[mathcalD[i]])
            
            # # Compute intersection and update constraint_ancestor
            # intersection = constraints_on_link[mathcalD[i]] & constraints_on_link[parent]
            # for constraint in intersection:
            #     constraint_ancestor[constraint, mathcalD[i]] = parent
            
            constraints_on_link[parent].update(constraints_on_link[mathcalD[i]])
        else:
            mathcalD[parent] += mathcalD[i]

    coupling_pairs = {}
    for n in mathcalN:
        coupling_pairs[n] = get_coupling_pairs(constraints_from_descendants[n])
    for con in constraints:
        con_id = con.constraint_index
        for link_idx, _ in con.links:
            index_now = link_idx
            while index_now in mathcalA and mathcalA[index_now] != 0:
                index_now = mathcalA[index_now]
                if con_id in coupling_pairs[index_now]:
                    if con_id in coupling_pairs[index_now][con_id]:
                        constraint_ancestor[con_id, link_idx] = index_now
                        while mathcalA[link_idx] != index_now:
                            constraint_ancestor[con_id, mathcalA[link_idx]] = index_now
                            link_idx = mathcalA[link_idx]
                        link_idx = index_now
                # index_now = mathcalA[index_now]
                        
    return K_direct, sorted(list(mathcalN)), mathcalA, mathcalD, constraints_from_descendants, constraint_ancestor

def PvOsimR(model, q, constraints):

    K_direct, mathcalN, mathcalA, mathcalD, constraints_from_descendants, constraint_ancestor = compute_metadata(model, constraints)

    
    Hi_A = list(model.Hi)
    P = {}
    Omega = {}
    Omega[(0,0)] = cs.SX(6, 6)

    for i in range(model.n_joints, 0, -1):
        parent = model.parents[i]
        Si = model.Si[i]
        Di = Si.T @ Hi_A[i] @ Si
        Omega[(parent, i)] = Si @ cs.inv(Di) @ Si.T

        if parent > 0:
            P[(parent, i)] = cs.SX.eye(6) - Hi_A[i] @ Omega[(parent, i)]
            Hi_A[parent] += P[(parent, i)] @ Hi_A[i]

            if mathcalD[i] != 0 and mathcalD[i] != i:
                P[(parent, mathcalD[i])] = P[(parent, i)] @ P[(i, mathcalD[i])]
        if mathcalD[i] != 0 and mathcalD[i] != i:
            Omega[(parent, mathcalD[i])] = Omega[(i, mathcalD[i])] + P[(i, mathcalD[i])].T @ Omega[(parent, i)] @ P[(i, mathcalD[i])]

    
    for i in mathcalN:
        if i in mathcalA:
            Omega[(0, i)] = Omega[(mathcalA[i], i)] + P[(mathcalA[i], i)].T @ Omega[(0, mathcalA[i])] @ P[(mathcalA[i], i)]

    for con in constraints:
        con_id = con.constraint_index
        con_links = []
        for link_idx, K_mat in con.links:
            con_links.append(link_idx)
        con_links.sort()
        for con_link in reversed(con_links):
            original_con_link = con_link
            while con_link in mathcalA:
                Anc = mathcalA[con_link]
                if con_id not in K_direct[Anc]:
                    K_direct[Anc][con_id] = cs.SX(K_mat.shape[0], 6)
                K_direct[Anc][con_id] += K_direct[con_link][con_id] @ P[(Anc, con_link)].T
                con_link = Anc
                if (con_id, original_con_link) in constraint_ancestor and constraint_ancestor[con_id, original_con_link] == con_link:
                    break
    

    Delassus = {}
    # initialize Delassus matrix with zeros
    for c1 in constraints:
        for c2 in constraints:
            Delassus[c1.constraint_index, c2.constraint_index] = cs.SX.zeros(c1.constraint_dim, c2.constraint_dim)

    for n in mathcalN:
        coupling_pairs = get_coupling_pairs(constraints_from_descendants[n])
        for c1 in coupling_pairs:
            for c2 in coupling_pairs[c1]:
                # Check constraint_ancestor for both constraints and this link
                omega_values = []
                if (c1, n) in constraint_ancestor:
                    omega_values.append(constraint_ancestor[c1, n])
                if (c2, n) in constraint_ancestor:
                    omega_values.append(constraint_ancestor[c2, n])
                
                if omega_values:
                    max_ancestor = max(omega_values)
                    compute_Omega(max_ancestor, n, Omega, mathcalA, P)
                    omega_matrix = Omega[(max_ancestor, n)]
                else:
                    omega_matrix = Omega[(0, n)]
                
                Delassus[c1, c2] += K_direct[n][c1] @ omega_matrix @ K_direct[n][c2].T
            
    for con in constraints:
        con_id = con.constraint_index
        for link_idx, K_mat in con.links:
            # Check constraint_ancestor for this constraint and link
            coupling_pairs = get_coupling_pairs(constraints_from_descendants[link_idx])
            if con_id in coupling_pairs:
                if con_id in coupling_pairs[con_id]:
                    continue
            
            if (con_id, link_idx) in constraint_ancestor:
                ancestor = constraint_ancestor[con_id, link_idx]
                if (ancestor, link_idx) not in Omega:
                    compute_Omega(ancestor, link_idx, Omega, mathcalA, P)
                omega_matrix = Omega[(ancestor, link_idx)]
            else:
                omega_matrix = Omega[(0, link_idx)]
            
            Delassus[con_id, con_id] += K_direct[link_idx][con_id] @ omega_matrix @ K_direct[link_idx][con_id].T
    
    return Delassus

def compute_Omega(ancestor, link_idx, Omega, mathcalA, P):
    """
    Compute Omega[(ancestor, link_idx)] when it doesn't exist.
    This function recursively computes the required Omega entry using the relationship:
    Omega[(ancestor, link_idx)] = Omega[(intermediate, link_idx)] + P[(intermediate, link_idx)].T @ Omega[(ancestor, intermediate)] @ P[(intermediate, link_idx)]
    """
    if (ancestor, link_idx) in Omega:
        return  # Already computed
    
    # Find the path from ancestor to link_idx through mathcalA
    if link_idx in mathcalA and mathcalA[link_idx] != 0:
        intermediate = mathcalA[link_idx]
        
        # Recursively ensure Omega[(ancestor, intermediate)] exists
        if (ancestor, intermediate) not in Omega:
            compute_Omega(ancestor, intermediate, Omega, mathcalA, P)
        
        # Compute Omega[(ancestor, link_idx)]
        if (intermediate, link_idx) in Omega and (ancestor, intermediate) in Omega:
            Omega[(ancestor, link_idx)] = (Omega[(intermediate, link_idx)] + 
                                          P[(intermediate, link_idx)].T @ 
                                          Omega[(ancestor, intermediate)] @ 
                                          P[(intermediate, link_idx)])
        else:
            # If we can't compute it, use a zero matrix as fallback
            import casadi as cs
            Omega[(ancestor, link_idx)] = cs.SX.zeros(6, 6)
    else:
        # If no path exists, use zero matrix
        import casadi as cs
        Omega[(ancestor, link_idx)] = cs.SX.zeros(6, 6)