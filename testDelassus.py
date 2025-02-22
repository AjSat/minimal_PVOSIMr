# Calls different Delassus algorithms and compares their outputs
import numpy as np
import PvOsim
from EFPA import EFPA
from copy import deepcopy
from PvOsimR import PvOsimR
import casadi as cs
import random

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

c2 = Constraint()
c2.K = np.random.randn(6,6)
c2.parent = 3
c2.constraint_index = 8

c3 = Constraint()
c3.K = np.random.randn(6,6)
c3.parent = 5
c3.constraint_index = 9

constraints.append(c1)
constraints.append(c2)
constraints.append(c3)

model_efpa = deepcopy(model)
model_pvosimr = deepcopy(model)

Delassus_PV = PvOsim.PvOsim(model, q_rand, constraints)
# print(Delassus_PV)
Delassus_efpa = EFPA(model_efpa, q_rand, constraints)
Delassus_pvosimr = PvOsimR(model, q_rand, constraints)

# compare the algorithms
for i in range(len(constraints)):
    for j in range(i, len(constraints)):
        assert(np.linalg.norm(cs.DM(Delassus_efpa[i,j]).full() - cs.DM(Delassus_pvosimr[i,j]).full()) < 1e-14)
        i_pv = model.constraints_supported[0][i] - model.n_joints - 1
        j_pv = model.constraints_supported[0][j] - model.n_joints - 1
        PV_mat = Delassus_PV[i*6:i*6 + 6, j*6 : j*6 + 6]
        assert(np.linalg.norm(cs.DM(PV_mat).full() - cs.DM(Delassus_pvosimr[i_pv,j_pv]).full()) < 1e-14)



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
    constraint_counter += 1
    constraints.append(c)

model_efpa = deepcopy(model)
model_pvosimr = deepcopy(model)

Delassus_PV = PvOsim.PvOsim(model, q_rand, constraints)
print(Delassus_PV)
Delassus_efpa = EFPA(model_efpa, q_rand, constraints)
Delassus_pvosimr = PvOsimR(model, q_rand, constraints)

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
        