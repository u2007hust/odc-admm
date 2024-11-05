import time
import gurobipy as gp
from gurobipy import GRB


def CentralizedSolver(c, H, h):
    print('='*30)
    print('centralized LP'.center(30))
    print('='*30)
    global obj_value
    N = c.shape[0]
    T = c.shape[1]
    k = c.shape[2]
    m = H.shape[2]
    cons = []
    try:
        start_time = time.time()
        model = gp.Model("centralized")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = 1000
        # create variables
        x = model.addMVar(shape=(N, T, k), lb=0, ub=1, vtype=GRB.CONTINUOUS)
        # set objective
        obj = 0
        for i in range(N):
            for t in range(T):
                obj += x[i, t, :] @ c[i, t, :]
        model.setObjective(obj, GRB.MINIMIZE)
        # add constraints
        for j in range(m):
            expr = 0
            for i in range(N):
                for t in range(T):
                    expr += (H[i, t, j, :] @ x[i, t, :] - h[i, j])
                    model.addConstr(sum(x[i, t, :]) <= 1)
            cons.append(model.addConstr(expr <= 0))
        model.optimize()
        obj_value = model.objVal
        dual = []
        for j in range(m):
            dual.append(cons[j].pi)
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    return x.x, obj_value, dual


