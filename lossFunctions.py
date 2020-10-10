import numpy as np

class Model():
    def __init__(self, d):
        self.w = np.ones(d)
        self.b = 1

    def mse(self, X, Y):
        """
        return MSE for given data points
        X: (N x d) ndarray
        y: (N,) ndarray
        """
        wx = X @ self.w
        wxb = wx + b
        diff = np.linalg.norm(y - wxb)
        return diff

    @property
    def w(self):
        return self.w

    @property
    def b(self):
        return self.b

class ols():
    """
    Ordinary Least squares regression
    """
    def __init__(self, d):
        super.__init__()

    def objective(self, X, Y):
        error = self.mse(X, Y)
        return error

    def gradientW(self, X):
        return X


class ridge():
    pass

class elastic():
    pass
