import numpy as np
import math
import gurobipy as gp
import copy
from gurobipy import GRB


def ODCADMM(tilde_p, tilde_c, b, bar_x, bar_y, Ni):
    print('=' * 30)
    print('decentralized QP by ODC-ADMM'.center(30))
    print('=' * 30)

    # number of mapping nodes
    J = tilde_c.shape[1]
    # number of data centers
    K = tilde_c.shape[2]
    # total number of nodes
    N = J + K
    # length of time horizon
    T = tilde_p.shape[0]
    # total number of constraints
    m = J + K

    # normalization
    b /= 100
    bar_y /= 100
    bar_x /= 100

    # primal solution y_k^{(t)}
    y = np.zeros((T, K))
    # primal solution x_{j,k}^{(t)}
    x = np.zeros((T, J, K))
    # current dual vector
    yy = np.zeros((J + K, J + K))
    # record of all dual vector
    yt = np.zeros((T, J + K, J + K))
    # vector p in the algorithm
    p = np.zeros((J + K, J + K))

    # communication matrix
    A = np.zeros((N, N))
    for i in range(N):
        di = len(Ni[i])
        A[i, i] = di * math.sqrt(1 + di)
        for j in Ni[i]:
            A[i, j] = -math.sqrt(1 + di)
    # parameters
    rho_default = 0.0001
    rho = rho_default
    alpha = calculateAlpha(rho, Ni, N, 1 / math.sqrt(T))

    # main loop
    for t in range(T):
        z = np.zeros((N, m))
        r = np.zeros((N, m))
        eta_t = 1 / math.sqrt(T)
        for i in range(N):
            di = len(Ni[i])
            z[i, :] += A[i, i] * yy[i, :] / (1 + di)
            for j in Ni[i]:
                z[i, :] += A[i, j] * yy[j, :] / (1 + di)
            p[i, :] += rho * z[i, :]
            # r is -gamma in the algorithm
            r[i, :] -= A[i, i] * (rho * z[i, :] + p[i, :])
            r[i, :] += alpha / eta_t * yy[i, :]
            for j in Ni[i]:
                r[i, :] -= A[j, i] * (rho * z[j, :] + p[j, :])
        for j in range(J):
            (x[t, j, :], ti) = getX(tilde_c[t, j, :], bar_x[j, :], b[t, j], r[j, :], alpha, eta_t, m, j)
            yy[j, :] = getYY_mapping_nodes(x[t, j, :], b[t, j], r[j, :], alpha, eta_t, m, j)
            yt[t, j, :] = copy.deepcopy(yy[j, :])
        for k in range(K):
            (y[t, k], ti) = getY(tilde_p[t, k], bar_y[k], r[k + J, :], alpha, eta_t, k)
            yy[k + J, :] = getYY_data_centers(y[t, k], r[k + J, :], alpha, eta_t, m, k)
            yt[t, k + J, :] = copy.deepcopy(yy[k + J, :])

    x *= 100
    y *= 100
    obj = 0
    for t in range(T):
        for k in range(K):
            obj += tilde_p[t, k] * y[t, k] * y[t, k]
            for j in range(J):
                obj += tilde_c[t, j, k] * x[t, j, k] * x[t, j, k]
    violation = np.zeros(m)
    capacity = np.zeros(m)
    for k in range(K):
        for t in range(T):
            for j in range(J):
                violation[k] += x[t, j, k]
            violation[k] -= y[t, k]
    for j in range(J):
        for t in range(T):
            for k in range(K):
                violation[j + K] -= x[t, j, k]
            violation[j + K] += b[t, j]
            capacity[j + K] += b[t, j]
    for i in range(len(violation)):
        if violation[i] < 0:
            violation[i] = 0

    return y, obj, violation, capacity


def calculateAlpha(rho, Ni, N, multiplier):
    max_value = 0
    for i in range(N):
        di = len(Ni[i])
        temp = rho * di * di * (1 + di) * multiplier
        for j in Ni[i]:
            dj = len(Ni[j])
            temp += rho * (1 + dj) * multiplier
        if temp > max_value:
            max_value = temp
    return 10 * rho + max_value


# compute {x_{j,1}^{(t)},...,x_{j, K}^{(t)}} for each mapping node j
def getX(tilde_c_j, bar_x_j, b_j_t, r_index, alpha, eta_t, m, j_index):
    # number of data centers
    K = bar_x_j.shape[0]
    try:
        # Create a new model
        model = gp.Model("getX")
        # model.Params.TimeLimit = 1000
        model.Params.OutputFlag = 0
        # Create variables
        x_j = []
        for k in range(K):
            x_j.append(model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS))
        ti = model.addMVar(shape=m, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
        # Auxiliary vector: $zi = H_i x_i - h_i - gamma_i - \tau_i$ to ensure Gurobi can identify the model as QP
        zi = []
        for i in range(m):
            zi.append(model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS))
        # Set objective
        obj = 0
        for k in range(K):
            obj += tilde_c_j[k] * x_j[k] * x_j[k]
        for i in range(m):
            obj += eta_t / alpha / 2 * zi[i] * zi[i]
        # obj += eta_t / alpha / 2 * (zi[:, 0] @ zi[:, 0])
        # obj += x_j[:, 0] @ x_j[:, 0] @ tilde_c_j
        model.setObjective(obj, GRB.MINIMIZE)
        # upper bound of x_j
        for k in range(K):
            model.addConstr(x_j[k] <= bar_x_j[k], "")
        # Add constraints
        model.addConstr(ti <= 0, "")
        # equality for zi
        for k in range(K):
            expr = r_index[k]
            expr += x_j[k]
            model.addConstr(expr - ti[k] == zi[k], "")
        for j in range(m - K):
            if j == j_index:
                expr = r_index[K + j]
                for k in range(K):
                    expr -= x_j[k]
                expr += b_j_t
                model.addConstr(expr - ti[j + K] == zi[j + K], "")
            else:
                model.addConstr(r_index[j + K] - ti[j + K] == zi[j + K], "")
        # Solve
        model.optimize()

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    x = np.zeros(K)
    for k in range(K):
        x[k] = x_j[k].x
    return x, ti.x


def getYY_mapping_nodes(x_j_t, b_j_t, r_index, alpha, eta_t, m, j_index):
    # number of data centers
    K = x_j_t.shape[0]
    # Compute z0
    z0 = np.zeros((m, 1))
    yi = np.zeros((m, 1))
    for k in range(K):
        z0[k] = r_index[k]
        z0[k] += x_j_t[k]
        if z0[k] >= 0:
            yi[k] = z0[k]
    for j in range(m - K):
        if j == j_index:
            z0[j + K] = r_index[j + K]
            for k in range(K):
                z0[j + K] -= x_j_t[k]
            z0[j + K] += b_j_t
        else:
            z0[j + K] = r_index[j + K]
    for j in range(m):
        if z0[j] >= 0:
            yi[j] = z0[j]

    return yi.reshape(1, -1) / alpha * eta_t


def getY(tilde_p_k_t, bar_y_k, r_index, alpha, eta_t, constraint_index):
    try:
        # Create a new model
        model = gp.Model("getY")
        # model.Params.TimeLimit = 1000
        model.Params.OutputFlag = 0
        # Create variables
        y_k = model.addVar(lb=0, ub=bar_y_k, vtype=GRB.CONTINUOUS)
        ti = model.addVar(lb=-GRB.INFINITY, ub=0, vtype=GRB.CONTINUOUS)
        # Auxiliary vector: $zi = H_i x_i - h_i - gamma_i - \tau_i$ to ensure Gurobi can identify the model as SOCP
        zi = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
        # Set objective
        obj = 0
        obj += tilde_p_k_t * y_k * y_k
        obj += eta_t / alpha / 2 * (zi * zi)
        model.setObjective(obj, GRB.MINIMIZE)
        # equality for zi
        expr = r_index[constraint_index]
        expr -= y_k
        model.addConstr(expr - ti == zi)
        # Solve
        model.optimize()

    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    return y_k.x, ti.x


def getYY_data_centers(y_k_t, r_index, alpha, eta_t, m, constraint_index):
    z0 = np.zeros((m, 1))
    yi = np.zeros((m, 1))
    for index in range(m):
        if index != constraint_index:
            z0[index] = r_index[index]
        else:
            z0[index] = r_index[index] - y_k_t
        if z0[index] >= 0:
            yi[index] = z0[index]

    return yi.reshape(1, -1) / alpha * eta_t
