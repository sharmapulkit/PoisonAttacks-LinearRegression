import numpy as np
import pandas as pd
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

    def mse(self, X, Y):
        """
        return MSE for given data points
        X: (N, d) ndarray
        y: (N,) ndarray
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.DataFrame):
            Y = Y.values

        wx = X @ self.w
        wxb = wx + self.b
        if not wxb.shape == Y.shape:
            wxb = wxb[:, None]
        loss = np.mean(np.square(wxb - Y))
        return loss

    def mse_at(self, wb, X, Y):
        """
        return MSE for given data points
        X: (N, d) ndarray
        y: (N,) ndarray
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.DataFrame):
            Y = Y.values

        wx = X @ wb[:-1]
        wxb = wx + wb[-1]
        loss = np.mean(np.square(wxb[:, None] - Y))
        return loss

    @abstractmethod
    def regularization(self, wb, X, Y):
        """
        Returns the regularization component of loss
        """
        pass

    @abstractmethod
    def regularization_at(self, wb, X, Y):
        """
        Returns the regularization component of loss
        """
        pass

    def objective(self, X, Y):
        """
        returns the objective function for the model
        """
        mse = self.mse(X, Y)
        regularization = self.regularization(X, Y)

        loss = mse + regularization
        return loss

    def objective_at(self, wb, X, Y):
        """
        returns the objective function for the model at given parameters
        """
        mse = self.mse_at(wb, X, Y)
        regularization = self.regularization_at(wb, X, Y)

        loss = mse + regularization
        return loss

    def mse_gradient(self, X, Y):
        """
        returns the gradient of model parameters
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.DataFrame):
            Y = Y.values

        wx = X @ self.w
        wxb = wx + self.b
        diff = wxb - Y
        grad_w = 2*np.mean(diff * X, axis=1)
        grad_b = 2*np.mean(diff)
        return np.concatenate((grad_w, grad_b), axis=None)


    def mse_gradient_at(self, wb, X, Y):
        """
        returns the gradient of model parameters at given parameters
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.DataFrame):
            Y = Y.values

        wx = X @ wb[:-1]
        wxb = wx + wb[-1]
        diff = wxb[:, None] - Y
        grad_w = 2*np.mean(diff * X, axis=0)
        grad_b = 2*np.mean(diff)
        return np.concatenate((grad_w, grad_b), axis=None)

    @abstractmethod
    def reg_gradient(self, X, Y):
        """
        returns the gradient of regularization loss at given parameters
        """
        pass

    def gradient(self, X, Y):
        """
        returns the gradient of the model objective 
        """
        mse_grad = self.mse_gradient(X, Y)
        reg_grad = self.reg_gradient(X, Y)
        total_grad = mse_grad + reg_grad

        return total_grad

    def gradient_at(self, wb, X, Y):
        """
        returns the gradient of model objective at the given model parameters
        """
        mse_grad = self.mse_gradient_at(wb, X, Y)
        reg_grad = self.reg_gradient_at(wb, X, Y)
        total_grad = mse_grad + reg_grad

        return total_grad

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

    def getParams(self):
        return np.concatenate([self._w, [self._b]])

    def setParams(self, params):
        self._w = params[:-1]
        self._b = params[-1]

    def fit(self, X, Y, max_iter=70):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.DataFrame):
            Y = Y.values

        theta0 = np.append(self.w, self.b)
        opts = {'maxiter':max_iter, 'disp':True}
        res = scipy.optimize.minimize(fun=self.objective_at, x0=theta0, jac=self.gradient_at, args=(X, Y), method='L-BFGS-B', options=opts)
        theta_star = res.x
        self.w = theta_star[:-1]
        self.b = theta_star[-1]

    def predict(self, X):
        wx = X @ self.w
        wxb = wx + self.b
        return wxb

    def score(self, X, Y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.DataFrame):
            Y = Y.values

        predictions = self.predict(X)
        mse = np.sum(np.square(Y - predictions[:, None]))
        return mse

class OLS(Model):
    """
    Ordinary Least squares regression
    """
    def __init__(self, d):
        super().__init__(d)

    def regularization(self, X, Y):
        return 0

    def regularization_at(self, wb, X, Y):
        return 0

    def reg_gradient(self, X, Y):
        return 0

    def reg_gradient_at(self, wb, X, Y):
        return 0

    def getG(self):
        return 0


class Ridge(Model):
    """
    Regression with L-2 regularization
    """
    def __init__(self, d, weight_decay=1e-3):
        super().__init__(d)
        self.weight_decay = weight_decay


    def regularization(self, X, Y):
        reg_loss = self.weight_decay*np.sum(np.square(self.w))
        return reg_loss

    def regularization_at(self, wb, X, Y):
        reg_loss = self.weight_decay*np.sum(np.square(wb[:-1]))
        return reg_loss

    def reg_gradient(self, X, Y):
        grad_w = 2*self.weight_decay*self.w
        grad_b = [0]
        reg_grad = np.concatenate((grad_w, grad_b), axis=None)
        return reg_grad

    def reg_gradient_at(self, wb, X, Y):
        grad_w = 2*self.weight_decay*wb[:-1]
        grad_b = [0]
        reg_grad = np.concatenate((grad_w, grad_b), axis=None)
        return reg_grad

    def getG(self):
        return np.eye(len(self.w))

class Lasso(Model):
    """
    Regression with L-1 regularization
    """
    def __init__(self, d, weight_decay=1e-3):
        super().__init__(d)
        self.weight_decay = weight_decay

    def regularization(self, X, Y):
        reg_loss = self.weight_decay*np.sum(np.abs(self.w))
        return reg_loss

    def regularization_at(self, wb, X, Y):
        reg_loss = self.weight_decay*np.sum(np.abs(wb[:-1]))
        return reg_loss

    def reg_gradient(self, X, Y):
        grad_w = self.weight_decay*np.where(self.w[:-1]>0, 1, -1)
        grad_b = [0]
        reg_grad = np.concatenate((grad_w, grad_b), axis=None)
        return reg_grad

    def reg_gradient_at(self, wb, X, Y):
        grad_w = self.weight_decay*np.where(wb[:-1]>0, 1, -1)
        grad_b = [0]
        reg_grad = np.concatenate((grad_w, grad_b), axis=None)
        return reg_grad

    def getG(self):
        return 0

class ElasticNet(Model):
    """
    Regression with both L-1 and L-2 regularization
    """
    def __init__(self, d, weight_decay_l1=1e-3, weight_decay_l2=1e-5):
        super().__init__(d)
        self.beta_1 = weight_decay_l1
        self.beta_2 = weight_decay_l2

    def regularization(self, X, Y):
        reg_loss = self.beta_1*np.sum(np.abs(self.w)) + self.beta_2*np.sum(np.square(self.w))
        return reg_loss

    def regularization_at(self, wb, X, Y):
        reg_loss = self.weight_decay*np.sum(np.abs(wb[:-1])) + self.beta_2*np.sum(np.square(wb[:-1]))
        return reg_loss

    def reg_gradient(self, X, Y):
        grad_w = self.beta_1*np.where(self.w[:-1]>0, 1, -1) + 2*self.weight_decay*self.w
        grad_b = [0]
        reg_grad = np.concatenate((grad_w, grad_b), axis=None)
        return reg_grad

    def reg_gradient_at(self, wb, X, Y):
        grad_w = self.weight_decay*np.where(wb[:-1]>0, 1, -1) + 2*self.weight_decay*wb[:-1]
        grad_b = [0]
        reg_grad = np.concatenate((grad_w, grad_b), axis=None)
        return reg_grad

    def getG(self):
        self.rho = 0.5
        return (1 - self.rho)*np.eye(len(self.w))
