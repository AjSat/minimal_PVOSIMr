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

# Test multiple binary constraints (each acting on 2 links)
print("Testing two binary constraints...")

n = 12
model = RandomKinematicTree(n)

# Create a tree with multiple branches
model.parents[4] = 2   # link 4 branches from link 2
model.parents[7] = 3   # link 7 branches from link 3  
model.parents[9] = 5   # link 9 branches from link 5

q_rand = np.random.randn(n, 1)

# Create multiple binary constraints
constraints = []
constraint_counter = n

# First binary constraint
c1 = Constraint()
c1.constraint_index = constraint_counter
c1.constraint_dim = 6
c1.links = [
    (2, np.random.randn(6, 6)),  # First link
    (6, np.random.randn(6, 6))   # Second link
]
constraints.append(c1)
constraint_counter += 1

# Second binary constraint
c2 = Constraint()
c2.constraint_index = constraint_counter
c2.constraint_dim = 4
c2.links = [
    (4, np.random.randn(4, 6)),  # First link
    (8, np.random.randn(4, 6))   # Second link
]
constraints.append(c2)
constraint_counter += 1

# Third binary constraint
c3 = Constraint()
c3.constraint_index = constraint_counter
c3.constraint_dim = 6
c3.links = [
    (4, np.random.randn(6, 6)),  # First link
    (8, np.random.randn(6, 6))   # Second link
]
constraints.append(c3)
constraint_counter += 1

# Test both loop-based algorithms
model_copy1 = deepcopy(model)
model_copy2 = deepcopy(model)

Delassus_pvosim_loops = PvOsimLoops(model_copy1, q_rand, constraints)
Delassus_pvosimr_loops = PvOsimRLoops(model_copy2, q_rand, constraints)

print(f"Testing {len(constraints)} binary constraints")
for i, c in enumerate(constraints):
    print(f"  Constraint {i+1}: links {c.links[0][0]} and {c.links[1][0]}, dimension {c.constraint_dim}")

# Compare all pairwise combinations of constraints
for i in range(len(constraints)):
    for j in range(i, len(constraints)):
        i_con_id = constraints[i].constraint_index
        j_con_id = constraints[j].constraint_index
        if i_con_id <= j_con_id:
            diff_norm = np.linalg.norm(
                cs.DM(Delassus_pvosim_loops[i_con_id, j_con_id]).full() - 
                cs.DM(Delassus_pvosimr_loops[i_con_id, j_con_id]).full()
            )
            assert diff_norm < 1e-14, f"Mismatch for constraints {i_con_id}, {j_con_id}: {diff_norm}"

print("Multiple binary constraints test passed!")

# Test random n-ary constraints with random tree structure
print("Testing random n-ary constraints...")

n = 20
model = RandomKinematicTree(n)

# # Create random tree structure with 3 branches
# number_branches = 3
# branch_points = []
# for i in range(number_branches):
#     # Choose a random child link (not link 1, which is root)
#     random_child_link = random.randint(2, model.n_joints)
#     # Choose a random parent link that comes before the child
#     random_parent_link = random.randint(1, random_child_link - 1)
#     model.parents[random_child_link] = random_parent_link
#     branch_points.append((random_child_link, random_parent_link))

# print(f"Created tree with branches: {branch_points}")

q_rand = np.random.randn(n, 1)

# Create 5 random n-ary constraints (arity between 2 and 4)
number_of_constraints = 5
constraint_counter = n
constraints = []

for i in range(number_of_constraints):
    c = Constraint()
    c.constraint_index = constraint_counter
    
    # Randomly choose the arity (number of links) between 2 and 4
    n_ary = random.randint(2,5)
    
    # Randomly select links for this constraint
    available_links = list(range(1, model.n_joints + 1))
    selected_links = random.sample(available_links, n_ary)
    
    # Random constraint dimension between 3 and 6
    constraint_dim = random.randint(3, 6)
    c.constraint_dim = constraint_dim
    
    # Create K matrices for each link in this constraint
    c.links = []
    for link_idx in selected_links:
        K_mat = np.random.randn(constraint_dim, 6)
        c.links.append((link_idx, K_mat))
    
    # Set parent to first link for compatibility
    c.parent = c.links[0][0]
    
    constraint_counter += 1
    constraints.append(c)
    
    print(f"  Constraint {i+1}: {n_ary}-ary constraint on links {[link[0] for link in c.links]}, dimension {constraint_dim}")

# Test both loop-based algorithms
model_copy1 = deepcopy(model)
model_copy2 = deepcopy(model)

Delassus_pvosim_loops = PvOsimLoops(model_copy1, q_rand, constraints)
Delassus_pvosimr_loops = PvOsimRLoops(model_copy2, q_rand, constraints)

print(f"Testing {len(constraints)} random n-ary constraints on {n}-link tree")

# Compare all pairwise combinations of constraints
max_diff = 0.0
for i in range(len(constraints)):
    for j in range(i, len(constraints)):
        i_con_id = constraints[i].constraint_index
        j_con_id = constraints[j].constraint_index
        
        if i_con_id <= j_con_id:
            diff_norm = np.linalg.norm(
                cs.DM(Delassus_pvosim_loops[i_con_id, j_con_id]).full() - 
                cs.DM(Delassus_pvosimr_loops[i_con_id, j_con_id]).full()
            )
            max_diff = max(max_diff, diff_norm)
            assert diff_norm < 1e-14, f"Mismatch for constraints {i_con_id}, {j_con_id}: {diff_norm}"

print(f"Random n-ary constraints test passed! (max difference: {max_diff:.2e})")