import numpy as np
import math
import gurobipy as gp
import copy
from gurobipy import GRB


def ODCADMM(c, H, h, Ni):
    print('=' * 30)
    print('decentralized LP'.center(30))
    print('=' * 30)
    # number of agents
    N = c.shape[0]
    # length of time horizon
    T = c.shape[1]
    # dimension of x_i^{(t)}
    k = c.shape[2]
    # dimension of constraints
    m = H.shape[2]

    # record of all primal solution
    x = np.zeros((N, T, k))
    # current dual vector of all agents
    y = np.zeros((N, m))
    # record of all dual solution
    yt = np.zeros((N, T, m))
    # vector p in the algorithm
    p = np.zeros((N, m))

    # communication matrix
    A = np.zeros((N, N))
    for i in range(N):
        di = len(Ni[i])
        A[i, i] = di * math.sqrt(1 + di)
        for j in Ni[i]:
            A[i, j] = -math.sqrt(1 + di)
    # parameters
    rho = 0.1
    alpha = calculateAlpha(rho, Ni, N, 1/math.sqrt(T))

    # main loop
    for t in range(T):
        z = np.zeros((N, m))
        r = np.zeros((N, m))
        eta_t = 1 / math.sqrt(T)
        for i in range(N):
            di = len(Ni[i])
            z[i, :] += A[i, i] * y[i, :] / (1 + di)
            for j in Ni[i]:
                z[i, :] += A[i, j] * y[j, :] / (1 + di)
            p[i, :] += rho * z[i, :]
            r[i, :] -= h[i]  # -h_i is added to gamma_i for convenience
            r[i, :] -= A[i, i] * (rho * z[i, :] + p[i, :])
            r[i, :] += alpha / eta_t * y[i, :]
            for j in Ni[i]:
                r[i, :] -= A[j, i] * (rho * z[j, :] + p[j, :])
        for i in range(N):
            (x[i, t, :], ti) = getX(c[i, t, :], H[i, t, :, :], r[i, :], alpha, eta_t)
            y[i, :] = getY(H[i, t, :, :], x[i, t, :], r[i, :], alpha, eta_t)
            yt[i, t, :] = copy.deepcopy(y[i, :])
    obj = 0
    s = np.zeros((N, T, k))
    for i in range(N):
        for t in range(T):
            s[i, t, :] = - c[i, t, :] - H[i, t, :, :].T @ yt[i, t, :]
            for j in range(c.shape[2]):
                if s[i, t, j] <= 0:
                    s[i, t, j] = 0
        for t in range(T):
            obj += max(s[i, t, :])
            obj += yt[i, t, :] @ (h[i] * np.ones(m))
    obj = 0
    for i in range(N):
        for t in range(T):
            obj += x[i, t, :] @ c[i, t, :]
    violation = 0
    capacity = 0
    for i in range(N):
        for t in range(T):
            violation += (H[i, t, :, :] @ x[i, t, :] - h[i])
            capacity += h[i]

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
    return 1 + max_value


def getX(cit, Hit, ri, alpha, eta_t):
    k = cit.shape[0]
    m = Hit.shape[0]
    # start_time = time.time()
    # Create a new model
    model = gp.Model("getX")
    # model.Params.TimeLimit = 1000
    model.Params.OutputFlag = 0
    # Create variables
    xi = model.addMVar(shape=k, lb=0, ub=1, vtype=GRB.CONTINUOUS)
    ti = model.addMVar(shape=(m, 1), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    zi = model.addMVar(shape=(m, 1), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
    # Set objective
    obj = 0
    obj += xi @ cit
    obj += eta_t / alpha / 2 * (zi[:, 0] @ zi[:, 0])
    model.setObjective(obj, GRB.MINIMIZE)
    # Add constraints
    model.addConstr(ti[:, 0] <= 0)
    model.addConstr(sum(xi) <= 1)
    for j in range(m):
        model.addConstr(Hit[j, :] @ xi + ri[j] - ti[j] == zi[j])
    # Solve
    model.optimize()

    return xi.x, ti.x


def getY(Hit, xit, ri, alpha, eta_t):
    m = Hit.shape[0]
    # Compute z0
    z0 = np.zeros((m, 1))
    yi = np.zeros((m, 1))
    for j in range(m):
        z0[j] += Hit[j, :] @ xit
        z0[j] += ri[j]
        if z0[j] >= 0:
            yi[j] = z0[j]
    return yi.reshape(1, -1) / alpha * eta_t
