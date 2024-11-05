import numpy as np


def graph_generator(N, p):
    if N == 1:
        return [[]]
    while 1:
        A = np.multiply((np.random.rand(N, N) <= p), 1)
        A = np.triu(A, 1) + np.triu(A, 1).T
        P = A
        temp = A
        for i in range(1, N):
            temp = np.matmul(temp, A)
            P = P + temp
            P = np.multiply(P > 0.5, 1)
        if (P == np.ones((N, N))).all():
            Ni = []
            for i in range(N):
                temp = []
                for j in range(N):
                    if A[i, j] > 0.5:
                        temp.append(j)
                Ni.append(temp)
            return Ni


def cycle_graph(N):
    Ni = []
    for i in range(N):
        temp = [i - 1, i + 1]
        if i == 0:
            temp = [N - 1, 1]
        if i == N - 1:
            temp = [N - 2, 0]
        Ni.append(temp)
    return Ni

if __name__ == '__main__':
    A = graph_generator(10, 0.2)
    A = cycle_graph(10)
    A
