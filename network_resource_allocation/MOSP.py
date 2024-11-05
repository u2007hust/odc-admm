import math

import numpy as np

def MOSP(tilde_p, tilde_c, b, bar_x, bar_y):
    print('=' * 30)
    print('decentralized QP by MOSP'.center(30))
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

    # parameters
    alpha = 0.05 / math.pow(T, 1/3)
    mu = 1000 * alpha
    # record of all solution x_t^{j,k}
    x = np.zeros((T, J, K))
    # record of all y_t^k
    y = np.zeros((T, J))
    # dual for constraints
    dual_K = np.zeros((K, 1))
    dual_J = np.zeros((J, 1))
    # main loop
    for t in range(1, T):
        # compute equation (40a)
        for j in range(J):
            for k in range(K):
                x[t, j, k] = x[t - 1, j, k] - alpha * tilde_c[t - 1, j, k] * 2 * x[t - 1, j, k] \
                             - alpha * (dual_K[k] - dual_J[j])
                if x[t, j, k] < 0:
                    x[t, j, k] = 0
                if x[t, j, k] > bar_x[j, k]:
                    x[t, j, k] = bar_x[j, k]
        # compute equation (40c)
        for k in range(K):
            y[t, k] = y[t - 1, k] - alpha * tilde_p[t - 1, k] * 2 * y[t - 1, k]
            y[t, k] += alpha * dual_K[k]
            if y[t, k] < 0:
                y[t, k] = 0
            if y[t, k] > bar_y[k]:
                y[t, k] = bar_y[k]
        # compute equation (40b)
        for j in range(J):
            dual_J[j] = dual_J[j] + mu * b[t, j]
            for k in range(K):
                dual_J[j] -= mu * x[t, j, k]
            if dual_J[j] < 0:
                dual_J[j] = 0
        # compute equation (40d)
        for k in range(K):
            dual_K[k] -= mu * y[t, k]
            for j in range(J):
                dual_K[k] += mu * x[t, j, k]
            if dual_K[k] < 0:
                dual_K[k] = 0

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