import time
import gurobipy as gp
from gurobipy import GRB


def CentralizedSolver(p, c, b, bar_x, bar_y):
    print('='*30)
    print('centralized QP'.center(30))
    print('='*30)

    global obj_value
    # number of mapping nodes
    J = c.shape[1]
    # number of data centers
    K = c.shape[2]
    # length of time horizon
    T = p.shape[0]

    cons = []
    try:
        start_time = time.time()
        model = gp.Model("centralized")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = 1000
        # create variables
        y = model.addMVar(shape=(T, K), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
        x = model.addMVar(shape=(T, J, K), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
        # set objective
        obj = 0
        for t in range(T):
            for k in range(K):
                obj += p[t, k] * y[t, k] * y[t, k]
                for j in range(J):
                    obj += x[t, j, k] * x[t, j, k] * c[t, j, k]
        model.setObjective(obj, GRB.MINIMIZE)
        # upper bounds of decision variables
        for t in range(T):
            for k in range(K):
                model.addConstr(y[t, k] <= bar_y[k])
                for j in range(J):
                    model.addConstr(x[t, j, k] <= bar_x[j, k])
        # add workload flow conservation constraints
        for k in range(K):
            expr = 0
            for t in range(T):
                for j in range(J):
                    expr += x[t, j, k]
                expr -= y[t, k]
            cons.append(model.addConstr(expr <= 0))
        for j in range(J):
            expr = 0
            for t in range(T):
                for k in range(K):
                    expr -= x[t, j, k]
                expr += b[t, j]
            cons.append(model.addConstr(expr <= 0))
        model.optimize()
        obj_value = model.objVal
        dual = []
        for cons_index in range(J + K):
            dual.append(cons[cons_index].pi)
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    return x.x, y.x, obj_value, dual


def CentralizedSolverPerSlot(p, c, b, bar_x, bar_y):
    print('='*30)
    print('centralized LP'.center(30))
    print('='*30)

    global obj_value
    # number of mapping nodes
    J = c.shape[0]
    # number of data centers
    K = c.shape[1]
    # length of time horizon
    T = p.shape[0]

    cons = []
    try:
        obj_value = 0
        for t in range(T):
            model = gp.Model("centralized")
            model.Params.OutputFlag = 0
            model.Params.TimeLimit = 1000
            # create variables
            y = model.addMVar(shape=K, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
            x = model.addMVar(shape=(J, K), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS)
            # set objective
            obj = 0
            for k in range(K):
                obj += p[t, k] * y[k] * y[k]
                for j in range(J):
                    obj += x[j, k] * x[j, k] * c[t, j, k]
            model.setObjective(obj, GRB.MINIMIZE)
            # upper bounds of decision variables
            for k in range(K):
                model.addConstr(y[k] <= bar_y[k])
                for j in range(J):
                    model.addConstr(x[j, k] <= bar_x[j, k])
            # add workload flow conservation constraints
            for k in range(K):
                expr = 0
                for j in range(J):
                    expr += x[j, k]
                expr -= y[k]
                cons.append(model.addConstr(expr <= 0))
            for j in range(J):
                expr = 0
                for k in range(K):
                    expr -= x[j, k]
                expr += b[t, j]
                cons.append(model.addConstr(expr <= 0))
            model.optimize()
            obj_value += model.objVal
        dual = []
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError:
        print('Encountered an attribute error')

    return x.x, y.x, obj_value, dual


