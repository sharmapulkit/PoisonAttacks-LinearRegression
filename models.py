import numpy as np
import load_datasets
import scipy.optimize

class Model():
    def __init__(self, d):
        self._w = np.ones(d)
        self._b = 1

    def mse(self, X, Y):
        """
        return MSE for given data points
        X: (N x d) ndarray
        y: (N,) ndarray
        """
        wx = X @ self._w
        wxb = wx + self._b
        diff = np.linalg.norm(Y - wxb)
        return diff

    def gradMSE(self, X, Y):
        wx = X @ self._w
        wxb = wx + self._b
        diff = Y - wxb
        grad = 2*np.sum(diff * X, axis=1)
        return grad

    def mse_at(self, wb, X, Y):
        """
        return MSE for given data points
        X: (N x d) ndarray
        y: (N,) ndarray
        """
        wx = X @ wb[:-1]
        wxb = wx + wb[-1]
        diff = np.linalg.norm(Y - wxb)
        return diff

    def gradMSE_at(self, wb, X, Y):
        wx = X @ wb[:-1]
        wxb = wx + wb[-1]
        diff = Y - wxb
        grad = 2*np.sum(diff * X, axis=1)
        return grad

    @property
    def w(self):
        self._w

    @w.setter
    def w(self, value):
        self._w = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = value

class ols(Model):
    """
    Ordinary Least squares regression
    """
    def __init__(self, d):
        super().__init__(d)

    def objective(self, X, Y):
        error = self.mse(X, Y)
        return error

    def gradientW(self, X, Y):
        error = self.gradMSE(X, Y)
        return error

    def objective_at(self, wb, X, Y):
        error = self.mse_at(wb, X, Y)
        return error

    def gradientW_at(self, wb, X, Y):
        error = self.gradMSE_at(wb, X, Y)
        return error

    def fit(self, data):
        X = data.X
        y = data.Y

        theta0 = np.append(self._w, self._b)
        opts = {'maxiter':30, 'disp':True}
        scipy.optimize.minimize(self.objective_at, theta0, args=(X, y), method='L-BFGS-B', options=opts)


class ridge(Model):
    pass

class elastic(Model):
    pass
