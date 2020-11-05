import numpy as np
import load_datasets
import scipy.optimize
from abc import ABC, abstractmethod

class Model(ABC):
    """
    Base class for linear regression model
    """
    def __init__(self, d):
        self._w = np.ones(d)
        self._b = 1
    
    @abstractmethod
    def objective(self, X, Y):
        """
        returns the objective function for the model
        """
        pass

    @abstractmethod
    def objective_at(self, wb, X, Y):
        """
        returns the objective function for the model at given parameters
        """
        pass

    @abstractmethod
    def gradient(self, X, Y):
        """
        returns the gradient of model parameters
        """
        pass

    @abstractmethod
    def gradient_at(self, wb, X, Y):
        """
        returns the gradient of model parameters at given parameters
        """
        pass

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        self._w = value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        self._b = value

class OLS(Model):
    """
    Ordinary Least squares regression
    """
    def __init__(self, d):
        super().__init__(d)

    def objective(self, X, Y):
        """
        return MSE for given data points
        X: (N, d) ndarray
        y: (N,) ndarray
        """
        wx = X @ self.w
        wxb = wx + self.b
        loss = np.mean(np.square(wxb - Y))
        return loss

    def objective_at(self, wb, X, Y):
        """
        return MSE for given data points
        X: (N, d) ndarray
        y: (N,) ndarray
        """
        wx = X @ wb[:-1]
        wxb = wx + wb[-1]
        loss = np.mean(np.square(wxb - Y))
        return loss

    def gradient(self, X, Y):
        wx = X @ self.w
        wxb = wx + self.b
        diff = wxb - Y
        grad_w = 2*np.sum(diff * X, axis=1)
        grad_b = 2*np.sum(diff)
        return np.concatenate((grad_w, grad_b), axis=None)
        
    def gradient_at(self, wb, X, Y):
        wx = X @ wb[:-1]
        wxb = wx + wb[-1]
        diff = wxb - Y
        grad_w = 2*np.sum(diff * X, axis=1)
        grad_b = 2*np.sum(diff)
        return np.concatenate((grad_w, grad_b), axis=None)

    def fit(self, data, max_iter=30):
        X = data.X
        y = data.Y

        theta0 = np.append(self.w, self.b)
        opts = {'maxiter':max_iter, 'disp':True}
        theta_star, _, _ = scipy.optimize.minimize(func=self.objective_at, x0=theta0, fprime=self.gradient_at, args=(X, y), method='L-BFGS-B', options=opts)
        self.w(theta_star[:-1])
        self.b(theta_star[-1])
        


class Ridge(Model):
    """
    Regression with L-2 regularization
    """
    def __init__(self, d, weight_decay=1e-3):
        super().__init__(d)
        self.weight_decay = weight_decay

    def objective(self, X, Y):
        wx = X @ self.w
        wxb = wx + self.b
        loss = np.mean(np.square(wxb - Y))
        loss += self.weight_decay*np.sum(np.square(self.w))
        return loss

    def gradient(self, X, Y):
        wx = X @ self.w
        wxb = wx + self.b
        diff = wxb - Y
        grad_w = 2*np.sum(diff * X, axis=1)
        grad_w += 2*self.weight_decay*self.w
        grad_b = 2*np.sum(diff)
        return np.concatenate((grad_w, grad_b), axis=None)

    def objective_at(self, wb, X, Y):
        wx = X @ wb[:-1]
        wxb = wx + wb[-1]
        loss = np.mean(np.square(wxb - Y))
        loss += self.weight_decay*np.sum(np.square(wb[:-1]))
        return loss

    def gradient_at(self, wb, X, Y):
        wx = X @ wb[:-1]
        wxb = wx + wb[-1]
        diff = wxb - Y
        grad_w = 2*np.sum(diff * X, axis=1)
        grad_w += 2*self.weight_decay*wb[:-1]
        grad_b = 2*np.sum(diff)
        return np.concatenate((grad_w, grad_b), axis=None)

    def fit(self, data):
        X = data.X
        y = data.Y

        theta0 = np.append(self.w, self.b)
        opts = {'maxiter':30, 'disp':True}
        theta_star, _, _ = scipy.optimize.minimize(func=self.objective_at, x0=theta0, fprime=self.gradient_at, args=(X, y), method='L-BFGS-B', options=opts)
        self.w(theta_star[:-1])
        self.b(theta_star[-1])

class Lasso(Model):
    """
    Regression with L-1 regularization
    """
    def __init__(self, d, weight_decay=1e-3):
        super().__init__(d)
        self.weight_decay = weight_decay

    def objective(self, X, Y):
        wx = X @ self.w
        wxb = wx + self.b
        loss = np.mean(np.square(wxb - Y))
        loss += self.weight_decay*np.sum(np.abs(self.w))
        return loss

    def gradient(self, X, Y):
        wx = X @ self.w
        wxb = wx + self.b
        diff = wxb - Y
        grad_w = 2*np.sum(diff * X, axis=1)
        grad_w += self.weight_decay*np.where(self.w[:-1]>0, 1, -1)
        grad_b = 2*np.sum(diff)
        return np.concatenate((grad_w, grad_b), axis=None)

    def objective_at(self, wb, X, Y):
        wx = X @ wb[:-1]
        wxb = wx + wb[-1]
        loss = np.mean(np.square(wxb - Y))
        loss += self.weight_decay*np.sum(np.abs(wb[:-1]))
        return loss

    def gradient_at(self, wb, X, Y):
        wx = X @ wb[:-1]
        wxb = wx + wb[-1]
        diff = wxb - Y
        grad_w = 2*np.sum(diff * X, axis=1)
        grad_w += self.weight_decay*np.where(self.w[:-1]>0, 1, -1)
        grad_b = 2*np.sum(diff)
        return np.concatenate((grad_w, grad_b), axis=None)

    def fit(self, data):
        X = data.X
        y = data.Y

        theta0 = np.append(self.w, self.b)
        opts = {'maxiter':30, 'disp':True}
        theta_star, _, _ = scipy.optimize.minimize(func=self.objective_at, x0=theta0, fprime=self.gradient_at, args=(X, y), method='L-BFGS-B', options=opts)
        self.w(theta_star[:-1])
        self.b(theta_star[-1])

class ElasticNet(Model):
    """
    Regression with both L-1 and L-2 regularization
    """
    def __init__(self, d, weight_decay_l1=1e-3, weight_decay_l2=1e-5):
        super().__init__(d)
        self.beta_1 = weight_decay_l1
        self.beta_2 = weight_decay_l2

    def objective(self, X, Y):
        wx = X @ self.w
        wxb = wx + self.b
        loss = np.mean(np.square(wxb - Y))
        loss += self.beta_1*np.sum(np.abs(self.w)) + self.beta_2*np.sum(np.square(self.w))
        return loss

    def gradient(self, X, Y):
        wx = X @ self.w
        wxb = wx + self.b
        diff = wxb - Y
        grad_w = 2*np.sum(diff * X, axis=1)
        grad_w += self.beta_1*np.where(self.w[:-1]>0, 1, -1) + 2*self.weight_decay*self.w
        grad_b = 2*np.sum(diff)
        return np.concatenate((grad_w, grad_b), axis=None)

    def objective_at(self, wb, X, Y):
        wx = X @ wb[:-1]
        wxb = wx + wb[-1]
        loss = np.mean(np.square(wxb - Y))
        loss += self.weight_decay*np.sum(np.abs(wb[:-1])) + self.beta_2*np.sum(np.square(wb[:-1]))
        return loss

    def gradient_at(self, wb, X, Y):
        wx = X @ wb[:-1]
        wxb = wx + wb[-1]
        diff = wxb - Y
        grad_w = 2*np.sum(diff * X, axis=1)
        grad_w += self.weight_decay*np.where(self.w[:-1]>0, 1, -1) + 2*self.weight_decay*wb[:-1]
        grad_b = 2*np.sum(diff)
        return np.concatenate((grad_w, grad_b), axis=None)

    def fit(self, data):
        X = data.X
        y = data.Y

        theta0 = np.append(self.w, self.b)
        opts = {'maxiter':30, 'disp':True}
        theta_star, _, _ = scipy.optimize.minimize(func=self.objective_at, x0=theta0, fprime=self.gradient_at, args=(X, y), method='L-BFGS-B', options=opts)
        self.w(theta_star[:-1])
        self.b(theta_star[-1])
