import time
from multiprocessing import Pool as ThreadPool

import numpy as np

from src.exper2_quad.CentralizedQPSolver import CentralizedQPSolver
from src.exper2_quad.ODCADMM import ODCADMM
from src.graph_generator import cycle_graph
from src.graph_generator import graph_generator


def StatCalculate(c, H, h, Ni, E):
    start_time = time.time()
    (x, obj0, dualValue) = CentralizedQPSolver(c, H, h, E)
    print('centralized QP running time: %.2f' % (time.time() - start_time))
    print("obj: ", obj0, " dualValue: ", dualValue)

    start_time = time.time()
    (y, obj, violation, x, capacity) = ODCADMM(c, H, h, Ni, E)
    print('decentralized QP running time: %.2f' % (time.time() - start_time))
    print("obj: ", obj, " violation: ", np.linalg.norm(violation), "last dual value", y)

    regret = obj - obj0
    relative_regret = regret / -obj0 * 100
    constraint_vio = np.linalg.norm(violation)
    relative_vio = constraint_vio / np.linalg.norm(capacity) * 100

    return regret, relative_regret, constraint_vio, relative_vio


def StatCalculate_wrapper(args):
    return StatCalculate(*args)


seed = 10
np.random.seed(seed)

k = 5
m = 4
N = 6

# generate graphs and parameters of distributions
# s_num denotes the number of distribution tried
# in the experiments in our paper, s_num = 1
s_num = 1
Ni_sets = []
tilde_Ui_sets = []
hi_sets = []
bar_Ui_sets = []
for s in range(s_num):
    bar_Ui = np.random.rand(N) * 0.5 + 0.5
    tilde_Ui = np.random.rand(N) * 1 / 3 + 5 / 3
    hi = np.random.rand(N, m) * 1 / 2 + 1 / 2
    bar_Ui_sets.append(bar_Ui)
    tilde_Ui_sets.append(tilde_Ui)
    hi_sets.append(hi)

graph_num = 4
Ni_sets.append(graph_generator(N, 1))
Ni_sets.append(graph_generator(N, 0.8))
Ni_sets.append(graph_generator(N, 0.5))
Ni_sets.append(cycle_graph(N))

# 5 trials for each distribution
loop_num = 5

# length of the time horizon
# for each point in figures, this parameter need to be modified
T = int(1000)

regret_sets = []
mean_regret_sets = []
constraint_vio_sets = []
mean_constraint_vio_sets = []
relative_regret_sets = []
relative_vio_sets = []
mean_relative_regret_sets = []
mean_relative_vio_sets = []

for g in range(graph_num):
    Ni = Ni_sets[g]

    regret = np.zeros(s_num * loop_num)
    constraint_vio = np.zeros(s_num * loop_num)
    relative_regret = np.zeros(s_num * loop_num)
    relative_vio = np.zeros(s_num * loop_num)

    np.random.seed(seed)
    for s in range(s_num):
        tilde_Ui = tilde_Ui_sets[s]
        bar_Ui = bar_Ui_sets[s]
        h = hi_sets[s]

        parameters_list = []
        for loop in range(loop_num):
            c = np.zeros((N, T, k))
            H = np.zeros((N, T, m, k))
            E = np.zeros((N, T, k, k))
            R = np.random.rand(N, T, k, k) * 0.5 + 0.5
            for i in range(N):
                for t in range(T):
                    E[i, t, :, :] = np.matmul(R[i, t, :, :].T, R[i, t, :, :])
            for i in range(N):
                # c is the negative of c in the paper
                c[i, :, :] = - np.random.rand(T, k) * bar_Ui[i]
                H[i, :, :, :] = np.random.rand(T, m, k) * tilde_Ui[i]
            parameters = (c, H, h, Ni, E)
            parameters_list.append(parameters)

        # # # multiprocessing # # #
        nthreads = loop_num
        pool = ThreadPool(nthreads)

        result = pool.map(StatCalculate_wrapper, parameters_list)

        for loop in range(loop_num):
            regret[s * loop_num + loop] = result[loop][0]
            relative_regret[s * loop_num + loop] = result[loop][1]
            constraint_vio[s * loop_num + loop] = result[loop][2]
            relative_vio[s * loop_num + loop] = result[loop][3]

    print("regret: ", regret)
    regret_sets.append(regret)
    print("average regret: ", np.mean(regret))
    mean_regret_sets.append(np.mean(regret))
    print("relative regret: ", relative_regret)
    relative_regret_sets.append(relative_regret)
    print("average relative regret: ", np.mean(relative_regret))
    mean_relative_regret_sets.append(np.mean(relative_regret))
    print("violation: ", constraint_vio)
    constraint_vio_sets.append(constraint_vio)
    print("average violation: ", np.mean(constraint_vio))
    mean_constraint_vio_sets.append(np.mean(constraint_vio))
    print("relative violation: ", relative_vio)
    relative_vio_sets.append(relative_vio)
    print("average relative violation: ", np.mean(relative_vio))
    mean_relative_vio_sets.append(np.mean(relative_vio))
    print("finished")

print("regret_sets: ", regret_sets)
print("constraint_vio_sets: ", constraint_vio_sets)
print("relative_regret_sets: ", relative_regret_sets)
print("relative_vio_sets: ", relative_vio_sets)
print("mean_regret_sets: ", mean_regret_sets)
print("mean_constraint_vio_sets: ", mean_constraint_vio_sets)
print("mean_relative_regret_sets: ", mean_relative_regret_sets)
print("mean_relative_vio_sets: ", mean_relative_vio_sets)
