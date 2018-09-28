class LPSolver(object):

    def __init__(self, c, A, b):
        """
        :param c: dim is nx1
        :param A: dim is mxn
        :param b: dim is mx1
        """

        def solve(self, x0, mu=10, eps=1e-3):
            """
            Solves min. c^Tx s.t. Ax<=b.
            :param x0: strictly feasible starting point
            :param eps: how well to approximate the solution
            :param mu: interior point step size
            :return: eps-suboptimal solution to the LP
            """
            t = 1
            while self.m / t > eps:
                _SD_optimal(t)
                t *= mu
            return self.x

        def _SD_optimal(t):
            """
            Solves min. t*c^Tx - sum(log(-a_ix+b)) using steepest descent
            (newton or gradient descent).
            :param t: positive scalar
            """


if __name__ == "__main__":
    # generate (as you like) c,A,b
    c = ..., A = ..., b = ...

    # verify you generated an interesting problem -
    #	feasible (non empty) and achieved (bounded bellow)
    # you can use other solver for this purpose (e.g. cvxpy, scipy.optimize.linprog,..)

    # find strictly feasible point
    initSolver = LPSolver(...)
    x0 = initSolver.solve(...)

    # solve the LP problem
    mySolver = LPSolver(c, A, b)
    x_star = mySolver.solve(x0)