import numpy as np
from src.network_resource_allocation.CentralizedSolver import CentralizedSolver, CentralizedSolverPerSlot
from src.network_resource_allocation.ODCADMM import ODCADMM
from src.network_resource_allocation.MOSP import MOSP

seed = 10
np.random.seed(seed)

# numbers of mapping nodes
J = 5
# numbers of data centers
K = 5
# length of time horizon
T = int(150000)


# index 0~(J-1) are for the mapping nodes; index J~(K+J-1) are for the data centers
def get_Ni(mapping_nodes_num, data_centers_num):
    Ni = []
    for j in range(mapping_nodes_num):
        temp = []
        for k in range(data_centers_num):
            temp.append(k + mapping_nodes_num)
        Ni.append(temp)
    for k in range(data_centers_num):
        temp = []
        for j in range(mapping_nodes_num):
            temp.append(j)
        Ni.append(temp)
    return Ni


bar_x = np.random.rand(J, K) * 90 + 10
bar_y = np.random.rand(K) * 100 + 100

# number of trails
loop_num = 1
for loop in range(loop_num):
    # randomly generate dataset
    p = np.random.rand(T, K) * 2 + 1
    b = np.random.rand(T, J) * 100 + 50
    c = np.random.rand(T, J, K) * 2 + 2
    # b = np.random.rand(T, J) * 0 + 100
    x, y, obj0, dualValue = CentralizedSolver(p, c, b, bar_x, bar_y)
    print("T = ", T)
    print("offline optimal obj: ", obj0)
    y, obj, violation, capacity = MOSP(p, c, b, bar_x, bar_y)
    print("MOSP obj: ", obj, "regret: ", obj - obj0, " violation: ", np.linalg.norm(violation), "capacity: ",
          np.linalg.norm(capacity), "relative regret: ", obj / obj0 - 1, "relative violation: ",
          np.linalg.norm(violation) / np.linalg.norm(capacity))
    y, obj, violation, capacity = ODCADMM(p, c, b, bar_x, bar_y, get_Ni(J, K))
    print("ODC-ADMM obj: ", obj, "regret: ", obj - obj0, " violation: ", np.linalg.norm(violation), "capacity: ",
          np.linalg.norm(capacity), "relative regret: ", obj / obj0 - 1, "relative violation: ",
          np.linalg.norm(violation) / np.linalg.norm(capacity))
