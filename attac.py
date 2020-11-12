import models
import load_datasets
import numpy as np
import pandas as pd

class attack():
    def __init__(self):
        pass

class BGD(attack):
    """
    Perform a BGD attack on a given model
    """
    def __init__(self, data, data_poison, max_iters, eta, line_search_epsilon, advModel):
        super().__init__()
        self.data_tr = data.getTrain() ## object of type dataset_struct
        self.data_val = data.getVal() ## object of type dataset_struct
        self.data_poison = data_poison.whole
        self._max_iters = max_iters
        self.eta = eta
        self.line_search_epsilon = line_search_epsilon
        self.advModel = advModel
        self.lambdaG = 0.2

    def computeM(self, xc, yc):
        """
        xc: Point to compute M for : [d,1]
        yc: Label to compute M for : [1]
        """
        weights = self.advModel.w
        pred = self.advModel.predict(xc)
        residual = pred - yc
        prod = xc[:, None] @ weights[:, None].T
        return prod + residual.values

    def covariance(self, data):
        """ Return covariance matrix """
        cov = data.X.T @ data.X / len(data.X)
        return cov

    def mean(self, data):
        """ Return mean vector """
        mu = np.mean(data.X, axis=0)
        return mu

    def computeGrad_x(self, xc, yc):
        """
        Return the gradient w.r.t. x of the adversary model
        """
        sigma = self.covariance(self.data_tr)
        mu = self.mean(self.data_tr)
        M = self.computeM(xc, yc)

        eq7lhs = np.bmat([[sigma + self.lambdaG*self.advModel.getG(), mu[:, None]], [mu[:, None].T, np.array([[1]])]] )
        eq7rhs = -(1/len(self.data_val.X))*np.bmat([[M, self.advModel.w[:, None]], [-xc[:, None].T, -np.array([[1]]) ] ]).T

        # print("lhs:", eq7lhs.shape)
        # print("rhs:", eq7rhs.shape)
        theta_grad, _, _, _ = np.linalg.lstsq(eq7lhs, eq7rhs, rcond=None)
        # print(theta_grad.shape)

        ### Compute w_grad
        res = (self.data_val.Y.iloc[:, 0] - self.advModel.predict(self.data_val.X)).values
        w_grad = np.sum(res * self.data_val.X.values.T, axis=1)
        w_grad = np.append(w_grad, np.sum(res*self.data_val.Y.values.T, axis=1))

        grad = theta_grad @ w_grad
        return grad.T

    def line_search(self, model, params, xc, yc):
        """
        Returns an optimal data point using the given model and the starting poison point
        model: Object of Model class - OLS, Ridge
        data: dataframe object of validation dataset
        params: set of params to evaluate model
        xc: initial poison point
        """
        model.setParams(params)
        objective_prev = -float('inf')

        iters = 0
        xc_new = xc
        eta = self.eta
        while (True):
            ## Compute Gradient
            objective_curr = model.objective(self.data_val.X, self.data_val.Y)
            grad = self.computeGrad_x(xc, yc)
            grad_wxc = np.array(grad[:-1])
            grad_bxc = np.array(grad[-1])
            ## update xc
            xc_new = xc_new + grad_wxc[:, 0] * eta
            yc_new = yc # yc_new + grad_wyc[:, 0] * eta
            ## break if no progress or convergence
            print(objective_curr)
            if (np.abs(objective_curr - objective_prev) < self.line_search_epsilon or (iters > 100)):
                break

            if (objective_curr < objective_prev):
                xc_new = xc_new - grad * eta
                break

            objective_prev = objective_curr
            iters += 1

        return xc_new, yc_new


    def _generatePoisonPoints(self, model, epsilon):
        """
        Returns generated poisson points using Algorithm 1 in paper
        model: Object of Model class - OLS, Ridge
        advmodel: Object of Model class - OLS, Ridge
        data_tr: Original Training dataset
        data_val: Original Validation dataset
        ini_poisonPts: Initial set of poison points
        epsilon: positive constant for terminating condition
        """
        i = 0
        dataUnionX = pd.concat([self.data_tr.X, self.data_poison.X])
        dataUnionY = pd.concat([self.data_tr.Y, self.data_poison.Y])
        print("SHAPES:", self.data_tr.X.shape, self.data_poison.X.shape)
        print("SHAPES:", self.data_tr.Y.shape, self.data_poison.Y.shape)
        model.fit(dataUnionX, dataUnionY)
        wPrev = 0
        wCurr = 0
        while (i < self._max_iters):
            if (i != 0 and np.abs(wCurr - wPrev) < epsilon):
                break

            wCurr = self.advModel.objective(self.data_val.X, self.data_val.Y)
            theta = model.getParams()
            for c in range(0, self.data_poison.getSize()):
                xc = self.data_poison.X.iloc[c]
                yc = self.data_poison.Y.iloc[c]
                x, y = self.line_search( self.advModel, theta, xc, yc ) # Line Search
                self.data_poison.X.iloc[c] = x
                self.data_poison.Y.iloc[c] = y
                model.fit(dataUnionX, dataUnionY)
                wPrev = wCurr
                wCurr = self.advModel.objective(self.data_val.X, self.data_val.Y)
            i += 1

    def set_advmodel(self, m):
        self.advmodel = m

    def set_model(self, m):
        self.model = m

    def generatePoisonPoints(self, baselinemodel):
        """ Return a set of poison points """
        epsilon = 0.01
        # trainData = load_datasets.houseData()
        # trainData.load()

        poisonPoints = self._generatePoisonPoints(baselinemodel, epsilon)

        return self.data_poison



