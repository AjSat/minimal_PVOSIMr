# Calls different Delassus algorithms and compares their outputs
import numpy as np
import PvOsim
from EFPA import EFPA
from copy import deepcopy
from PvOsimR import PvOsimR
from PvOsim_loops import PvOsim as PvOsimLoops
from PvOsimR_loops import PvOsimR as PvOsimRLoops
import casadi as cs
import random
from copy import deepcopy

class RandomKinematicTree:

    def __init__(self, n):
        
        Hi = []
        Si = []
        for i in range(n):
            rand_M = np.random.randn(6,6)
            Hi.append(rand_M.T @ rand_M) # might not be a physically consistent inertia. Atleast positive definite (in all likelihood).
            Si.append(np.random.randn(6,1))
        
        self.n_joints = n - 1
        self.Hi = Hi
        self.Si = Si
        self.Hi_A = [None]*n
        self.Ki_A = [None]*n
        self.Li_A = [None]*n
        self.Di =  [None]*n
        self.Di_inv = [None]*n
        self.constraints_supported = [None]*n
        self.parents = [None]*n
        self.Pi = [None]*n

        for i in range(1, n):
            self.parents[i] = i-1

class Constraint:

    def __init__(self):
        pass

n = 7
model = RandomKinematicTree(n)
model.parents[6] = 2
q_rand = np.random.randn(n, 1)

constraints = []
c1 = Constraint()
c1.K = np.random.randn(6,6)
c1.parent = 6
c1.constraint_index = 7
c1.constraint_dim = 6
c1.links = [(c1.parent, c1.K)]

c2 = Constraint()
c2.K = np.random.randn(6,6)
c2.parent = 3
c2.constraint_index = 8
c2.constraint_dim = 6
c2.links = [(c2.parent, c2.K)]

c3 = Constraint()
c3.K = np.random.randn(6,6)
c3.parent = 5
c3.constraint_index = 9
c3.constraint_dim = 6
c3.links = [(c3.parent, c3.K)]

constraints.append(c1)
constraints.append(c2)
constraints.append(c3)

model_efpa = deepcopy(model)
model_pvosimr = deepcopy(model)

Delassus_PV = PvOsim.PvOsim(model, q_rand, constraints)
# print(Delassus_PV)
Delassus_efpa = EFPA(model_efpa, q_rand, constraints)
Delassus_pvosimr = PvOsimR(model, q_rand, constraints)

model_loops = deepcopy(model)
Delassus_pvosim_loops = PvOsimLoops(model_loops, q_rand, constraints)
Delassus_pvosimr_loops = PvOsimRLoops(model_loops, q_rand, constraints)

# compare the algorithms
for i in range(len(constraints)):
    for j in range(i, len(constraints)):
        assert(np.linalg.norm(cs.DM(Delassus_efpa[i,j]).full() - cs.DM(Delassus_pvosimr[i,j]).full()) < 1e-14)
        i_pv = model.constraints_supported[0][i] - model.n_joints - 1
        j_pv = model.constraints_supported[0][j] - model.n_joints - 1
        PV_mat = Delassus_PV[i*6:i*6 + 6, j*6 : j*6 + 6]
        assert(np.linalg.norm(cs.DM(PV_mat).full() - cs.DM(Delassus_pvosimr[i_pv,j_pv]).full()) < 1e-14)

        i_con_id = model.constraints_supported[0][i]
        j_con_id = model.constraints_supported[0][j]
        assert(np.linalg.norm(cs.DM(Delassus_pvosimr[i_pv, j_pv]).full() - cs.DM(Delassus_pvosim_loops[i_con_id, j_con_id]).full()) < 1e-14)

        if i_con_id <= j_con_id:
            assert(np.linalg.norm(cs.DM(Delassus_pvosimr[i_pv, j_pv]).full() - cs.DM(Delassus_pvosim_loops[i_con_id, j_con_id]).full()) < 1e-14)



n = 20
model = RandomKinematicTree(n)

number_branches_approx = 5
for i in range(number_branches_approx):
    random_child_link = random.randint(2, model.n_joints)
    random_parent_link = random.randint(1, random_child_link - 1)
    model.parents[random_child_link] = random_parent_link
# model.parents[6] = 2
q_rand = np.random.randn(n, 1)

number_of_constraints = 10
constraint_counter = n
constraints = []
for i in range(number_of_constraints):
    c = Constraint()
    c.K = np.random.randn(6,6)
    c.parent = random.randint(1, model.n_joints)
    c.constraint_index = constraint_counter
    c.links = [(c.parent, c.K)]
    c.constraint_dim = 6
    constraint_counter += 1
    constraints.append(c)

model_efpa = deepcopy(model)
model_pvosimr = deepcopy(model)

Delassus_PV = PvOsim.PvOsim(model, q_rand, constraints)
print(Delassus_PV)
Delassus_efpa = EFPA(model_efpa, q_rand, constraints)
Delassus_pvosimr = PvOsimR(model, q_rand, constraints)

model_copy = deepcopy(model)
Delassus_pvosim_loops = PvOsimLoops(model_copy, q_rand, constraints)
model_copy = deepcopy(model)
Delassus_pvosimr_loops = PvOsimRLoops(model_copy, q_rand, constraints)


print(Delassus_efpa)

# compare the algorithms
for i in range(len(constraints)):
    for j in range(i, len(constraints)):
        assert(np.linalg.norm(cs.DM(Delassus_efpa[i,j]).full() - cs.DM(Delassus_pvosimr[i,j]).full()) < 1e-14)
        i_pv = model.constraints_supported[0][i] - model.n_joints - 1
        j_pv = model.constraints_supported[0][j] - model.n_joints - 1

        PV_mat = Delassus_PV[i*6:i*6 + 6, j*6 : j*6 + 6]
        if i_pv > j_pv:
            temp = i_pv
            i_pv = j_pv 
            j_pv = temp
            assert(np.linalg.norm(cs.DM(PV_mat).full() - cs.DM(Delassus_pvosimr[i_pv,j_pv].T).full()) < 1e-14)
        else:
            assert(np.linalg.norm(cs.DM(PV_mat).full() - cs.DM(Delassus_pvosimr[i_pv,j_pv]).full()) < 1e-14)

        i_con_id = model.constraints_supported[0][i]
        j_con_id = model.constraints_supported[0][j]
        if i_con_id <= j_con_id:
            assert(np.linalg.norm(cs.DM(Delassus_pvosimr[i_pv, j_pv]).full() - cs.DM(Delassus_pvosim_loops[i_con_id, j_con_id]).full()) < 1e-14)
            assert(np.linalg.norm(cs.DM(Delassus_pvosim_loops[i_con_id, j_con_id]).full() - cs.DM(Delassus_pvosimr_loops[i_con_id, j_con_id]).full()) < 1e-14)

# Test single binary constraint (acting on 2 links)
print("Testing single binary constraint...")

n = 10
model = RandomKinematicTree(n)

# Create a simple tree with one branch
model.parents[5] = 2  # link 5 branches from link 2

q_rand = np.random.randn(n, 1)

# Create a single binary constraint acting on 2 links
constraints = []
c = Constraint()
c.constraint_index = n
c.constraint_dim = 6
# c.K = np.random.randn(6, 6)  # 3D constraint
c.parent = 3  # For compatibility with existing code
c.links = [
    (3, np.random.randn(6, 6)),  # First link
    (7, np.random.randn(6, 6))   # Second link
]
constraints.append(c)

# Test both loop-based algorithms
model_copy1 = deepcopy(model)
model_copy2 = deepcopy(model)

Delassus_pvosim_loops = PvOsimLoops(model_copy1, q_rand, constraints)
Delassus_pvosimr_loops = PvOsimRLoops(model_copy2, q_rand, constraints)

print(f"Testing binary constraint acting on links {c.links[0][0]} and {c.links[1][0]}")

# Compare the results - for single constraint, just check diagonal element
con_id = c.constraint_index
diff_norm = np.linalg.norm(
    cs.DM(Delassus_pvosim_loops[con_id, con_id]).full() - 
    cs.DM(Delassus_pvosimr_loops[con_id, con_id]).full()
)
assert diff_norm < 1e-14, f"Mismatch for binary constraint {con_id}: {diff_norm}"

print("Single binary constraint test passed!")

