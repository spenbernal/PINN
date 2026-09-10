import numpy as np

class FiniteDifferenceHeatEq:
    def solve_dirichlet(self, x: np.ndarray, t: np.ndarray, s, init_cond):
        u = np.zeros((len(t), len(x)))
        u[0, :] = init_cond
        u[:, 0] = 0
        u[:, -1] = 0

        J = len(x) - 1
        for n in range(0, len(u) - 1):
            u[n+1, 1:J] = s*(u[n, 2:J+1] + u[n, 0:J-1]) + (1-2*s)*u[n,1:J]

        return u
