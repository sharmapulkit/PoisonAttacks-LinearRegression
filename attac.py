import models
import load_datasets
import numpy as np
import pandas as pd
import time

from multiprocessing import Pool
from scipy.optimize import line_search

class attack():
    def __init__(self):
        pass

class BGD(attack):
    """
    Perform a BGD attack on a given model
    """
    def __init__(self, data, data_poison, max_iters, eta, line_search_epsilon, advModel, rvo=False):
        super().__init__()
        self.data_tr = data.getTrain() ## object of type dataset_struct
        self.data_val = data.getVal() ## object of type dataset_struct
        self.data_poison = data_poison.whole
        self._max_iters = max_iters
        self.eta = eta
        self.line_search_epsilon = line_search_epsilon
        self.advModel = advModel
        self.lambdaG = 0.2
        self.rvo = rvo

        self._mean = None
        self._covariance = None
        self.eq7lhs = None

    def computeM(self, xc, yc):
        """
        xc: Point to compute M for : [d,1]
        yc: Label to compute M for : [1]
        """
        weights = self.advModel.w
        pred = self.advModel.predict(xc)
        residual = pred - yc
        prod = xc[:, None] @ weights[:, None].T
        return prod + residual

    def covariance(self, data):
        """ Return covariance matrix """
        if self._covariance is None:
            self._covariance = data.X.T @ data.X / len(data.X)
        return self._covariance

    def mean(self, data):
        """ Return mean vector """
        if self._mean is None:
            self._mean = np.mean(data.X, axis=0)
        return self._mean

    def computeGrad_x(self, xc, yc, valData=False):
        """
        Return the gradient w.r.t. x of the adversary model
        """
        sigma = self.covariance(self.data_tr)
        mu = self.mean(self.data_tr)
        M = self.computeM(xc, yc)

        dataset = self.data_tr
        if (valData is True):
            dataset = self.data_val

        if self.eq7lhs is None:
            self.eq7lhs = np.bmat([[sigma + self.lambdaG*self.advModel.getG(), mu[:, None]], [mu[:, None].T, np.array([[1]])]] )

        # dim = len(xc)
        # np.random.seed(0)
        # xc, yc, eq7lhs, mu, M = \
        #     np.random.random((dim)), np.random.random((1, 1)), np.random.random((dim + 1, dim + 1)), np.random.random((dim, 1)), np.random.random((dim, dim))

        if (self.rvo is True):
            # eq7rhs = -(1/len(dataset.X))*np.bmat([[M, self.advModel.w[:, None]], [-xc[:, None].T, -np.array([[1]]) ] ]).T
            eq7rhs = -(1/len(dataset.X))*np.bmat([[M, -xc[:, None]], [self.advModel.w[:, None].T, -np.array([[1]]) ] ]).T
        else:
            eq7rhs = -(1/len(dataset.X))*np.bmat([[M, self.advModel.w[:, None]]]).T

        # print("lhs:", eq7lhs.shape)
        # print("rhs:", eq7rhs.shape)
        theta_grad, _, _, _ = np.linalg.lstsq(self.eq7lhs, eq7rhs, rcond=None)
        # print("Attack:", theta_grad)

        ### Compute w_grad
        res = (self.advModel.predict(dataset.X) - dataset.Y[:, 0])
        obj_grad = np.sum(res * dataset.X.T, axis=1)
        obj_grad = np.append(obj_grad, dataset.Y.shape[0]*np.sum(res, axis=0))[:, None]
        # print("Residual shape:", np.bmat([[M, self.advModel.w[:, None]]]).T.shape)

        if (self.rvo is False):
            wb_grad_yc = np.zeros((len(mu) + 1, 1))
            theta_grad = np.append(theta_grad, wb_grad_yc, axis=1 )

        ### grad_xc = np.concatenate((obj_grad[:-1, None].T @ w_grad_xc, ))
        grad = theta_grad.T @ obj_grad / dataset.Y.shape[0]
        return grad

    def line_search(self, model, params, xc, yc, valData=False):
        """
        Returns an optimal data point using the given model and the starting poison point
        model: Object of Model class - OLS, Ridge
        data: dataframe object of validation dataset
        params: set of params to evaluate model
        xc: initial poison point
        """
        dataset = self.data_tr
        if (valData is True):
            dataset = self.data_val

        new_model = models.Ridge(xc.shape[0])
        new_model.setParams(params)
        objective_prev = -float('inf')

        iters = 0
        xc_new = xc
        yc_new = yc
        eta = self.eta
        beta = 0.08
        taintedTr = load_datasets.dataset_struct(np.append(np.copy(self.data_tr.X), xc_new[:, None].T, axis=0), np.append(np.copy(self.data_tr.Y), yc_new[:, None].T, axis=0) )
        new_model.fit(taintedTr.X, taintedTr.Y)
        objective_prev = new_model.mse(taintedTr.X, taintedTr.Y)

        grad = self.computeGrad_x(xc, yc)
        grad_xc = np.squeeze(np.array(grad[:-1]))
        grad_yc = np.squeeze(np.array(grad[-1]))
        # print("grad_yc:", grad_yc)
        while (True):
            ## Compute Gradient
            ## update xc
            xc_new = xc_new + grad_xc * eta
            xc_new = np.clip(xc_new, 0, 1)
            
            if (self.rvo is True):
                yc_new += grad_yc * eta
                yc_new = np.minimum(np.ones(yc_new.shape), np.maximum(np.zeros_like(yc_new), yc_new))

            taintedTr.X[-1] = xc_new
            taintedTr.Y[-1] = yc_new
            new_model.fit(taintedTr.X, taintedTr.Y, max_iter=400)

            objective_curr = new_model.mse(taintedTr.X, taintedTr.Y)
            ## break if no progress or convergence
            if (np.abs(objective_curr - objective_prev) < self.line_search_epsilon or (iters > 8)):
                # print("objective_curr")
                break

            if (objective_curr < objective_prev and iters > 0):
                xc_new = xc_new - grad_xc * eta
                if (self.rvo is True):
                    yc_new = yc_new - grad_yc * eta
                break
                
            if (iters > 0):
                eta = eta*beta
            
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
        dataUnionX = np.concatenate((self.data_tr.X, self.data_poison.X))
        dataUnionY = np.concatenate((self.data_tr.Y, self.data_poison.Y))
        print("SHAPES:", self.data_tr.X.shape, self.data_poison.X.shape)
        print("SHAPES:", self.data_tr.Y.shape, self.data_poison.Y.shape)
        model.fit(dataUnionX, dataUnionY)
        self.advModel.setParams(model.getParams())
        bestParams = model.getParams()
        wPrev = self.advModel.mse(self.data_val.X, self.data_val.Y)
        wCurr = 0

        wBest = wPrev
        while (i < self._max_iters):
            print("Poisoning Iter:", i)
            if (i >= 18 and np.abs(wCurr - wPrev) < epsilon): ## Minimum number of iterations
                print("Current, Prev:", wCurr, wPrev)
                break

            wPrev = wCurr
            wCurr = self.advModel.mse(self.data_val.X, self.data_val.Y)
            theta = model.getParams()
            for c in range(0, self.data_poison.getSize()):
                xc = self.data_poison.X[c]
                yc = self.data_poison.Y[c]
                x, y = self.line_search( self.advModel, theta, xc, yc ) # Line Search
                dataUnionX[self.data_tr.getSize() + c] = x
                dataUnionY[self.data_tr.getSize() + c] = y
                # time_after_assigning = time.time()
                # print("Time to assign:", time_after_assigning - time_after_line_search)
                model.fit(dataUnionX, dataUnionY)
                # time_after_training_model = time.time()
                # print("Time to train:", time_after_training_model - time_after_assigning)
                self.advModel.setParams(model.getParams())
                wCurr = self.advModel.mse(self.data_val.X, self.data_val.Y)

            i += 1
            if (wCurr <= wPrev):
                self.eta *= 0.75
                self.advModel.setParams(bestParams)                
                for c in range(0, self.data_poison.getSize()):
                    dataUnionX[self.data_tr.getSize() + c] = self.data_poison.X[c]
                    dataUnionY[self.data_tr.getSize() + c] = self.data_poison.Y[c]
            
            if (wCurr > wBest):
                wBest = wCurr
                bestParams = model.getParams()
                for c in range(0, self.data_poison.getSize()):
                    self.data_poison.X[c] = dataUnionX[self.data_tr.getSize() + c]
                    self.data_poison.Y[c] = dataUnionY[self.data_tr.getSize() + c]
            
            print("Current loss:", self.advModel.mse(self.data_val.X, self.data_val.Y))


    def line_search_tuple(self, in_tuple):
        return self.line_search(*in_tuple)

    def _generatePoisonPoints_Pool1(self, model, epsilon):
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
        self.advModel.setParams(model.getParams())
        bestParams = model.getParams()
        wPrev = self.advModel.objective(self.data_val.X, self.data_val.Y)
        wCurr = 0

        wBest = wPrev
        while (i < self._max_iters):
            print("Poisoning Iter:", i)
            if (i != 0 and np.abs(wCurr - wPrev) < epsilon):
                print("Current, Prev:", wCurr, wPrev)
                break

            wPrev = wCurr
            wCurr = self.advModel.objective(self.data_val.X, self.data_val.Y)
            theta = model.getParams()

            #### Pool
            workerpool = Pool(max(1, 10))
            args = [(self.advModel, theta, self.data_poison.X[c], self.data_poison.Y[c]) for c in range(0, self.data_poison.getSize())]
            
            for i, cur_pois_res in enumerate(workerpool.map(self.line_search_tuple, args)):
                dataUnionX[self.data_tr.getSize() + i] = cur_pois_res[0]
                dataUnionY[self.data_tr.getSize() + i] = cur_pois_res[1]
                model.fit(dataUnionX, dataUnionY)
                self.advModel.setParams(model.getParams())
                wCurr = self.advModel.objective(self.data_val.X, self.data_val.Y)

            # for c in range(0, self.data_poison.getSize()):
            #     xc = self.data_poison.X.iloc[c]
            #     yc = self.data_poison.Y.iloc[c]
            #     x, y = self.line_search( self.advModel, theta, xc, yc ) # Line Search
            #     dataUnionX.iloc[self.data_tr.getSize() + c] = x
            #     dataUnionY.iloc[self.data_tr.getSize() + c] = y
            #     model.fit(dataUnionX, dataUnionY)
            #     self.advModel.setParams(model.getParams())
            #     wCurr = self.advModel.objective(self.data_val.X, self.data_val.Y)

            i += 1
            if (wCurr < wPrev):
                self.eta *= 0.75
                self.advModel.setParams(bestParams)                
                for c in range(0, self.data_poison.getSize()):
                    dataUnionX[self.data_tr.getSize() + c] = self.data_poison.X[c]
                    dataUnionY[self.data_tr.getSize() + c] = self.data_poison.Y[c]
            
            if (wCurr > wBest):
                wBest = wCurr
                bestParams = model.getParams()
                for c in range(0, self.data_poison.getSize()):
                    self.data_poison.X[c] = dataUnionX[self.data_tr.getSize() + c]
                    self.data_poison.Y[c] = dataUnionY[self.data_tr.getSize() + c]


            # print("Current loss:", model.mse(self.data_val.X, self.data_val.Y))
            print("Current loss:", self.advModel.mse(self.data_val.X, self.data_val.Y), wCurr)

    def set_advmodel(self, m):
        self.advmodel = m

    def set_model(self, m):
        self.model = m

    def generatePoisonPoints(self, baselinemodel):
        """ Return a set of poison points """
        epsilon = 1e-3 ## From the paper
        # trainData = load_datasets.houseData()
        # trainData.load()

        self._generatePoisonPoints(baselinemodel, epsilon)
        mse = baselinemodel.objective(np.concatenate((self.data_tr.X, self.data_poison.X)), np.concatenate((self.data_tr.Y, self.data_poison.Y)))
        print("final MSE Train:", mse)

        return self.data_poison, mse
