import pandas as pd
import sys
import os
import numpy as np

class dataset_struct:
    def __init__(self, X, Y):
        self._X = X
        self._Y = Y

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    def getSize(self):
        return len(self._X)

class datasets():
    def __init__(self, train_frac):
        self.train_frac = train_frac
        self.train = None
        self.val = None
        self.whole = None

    def load(self):
        self.df = pd.read_csv(self.dataPath)
        feature_columns = self.df.columns.drop(self.target_columns)

        X = self.df[feature_columns].values
        Y = self.df[self.target_columns].values
        self.whole = dataset_struct(X, Y)

        N = len(X)

        all_idxs = np.arange(N)
        np.random.shuffle(all_idxs)
        train_idxs = all_idxs[:int(self.train_frac*N)]
        val_idxs = all_idxs[int(self.train_frac*N):]

        self.train = dataset_struct(X[train_idxs], Y[train_idxs])
        self.val = dataset_struct(X[val_idxs], Y[val_idxs])

    def get_df(self):
        return self.df

    def getSize(self):
        return len(self.df)

    def getTrain(self):
        if (self.train is None):
            raise ValueError("Data not loaded")
        return self.train

    def getVal(self):
        if (self.val is None):
            raise ValueError("Val data not loaded")
        return self.val

class houseData(datasets):
    def __init__(self, train_frac=0.7):
        super().__init__(train_frac)
        self.dataPath = "./datasets/house-processed.csv"
        self.target_columns = ['SalePrice']

class loanData(datasets):
    def __init__(self, train_frac=0.7):
        super().__init__(train_frac)
        self.dataPath = "./datasets/loan-processed.csv"
        self.target_columns = ['int_rate']

class pharmpreprocData(datasets):
    def __init__(self, train_frac=0.7):
        super().__init__(train_frac)
        self.dataPath = "./datasets/pharm-preproc.csv"
        self.target_columns = ['TherapeuticDoseofWarfarin']


class initialDataSet(datasets):
    def __init__(self):
        pass

    def loadInvFlip(self, data_tr, N):
        """
        Load an Inverse Flip data set randomly from training data
        """
        trainDataSize = data_tr.getSize()
        assert N < trainDataSize

        arr = range(0, trainDataSize)
        selectedIdxs = np.random.choice(arr, size=N, replace=False)

        self.X = data_tr.whole.X[selectedIdxs]
        self.Y = 1 - data_tr.whole.Y[selectedIdxs]

        self.whole = dataset_struct(self.X, self.Y)


    def loadBFlip(self, data_tr, N):
        """
        Load a Boundary Flip data set randomly from training data
        """
        trainDataSize = data_tr.getSize()
        assert N < trainDataSize

        arr = range(0, trainDataSize)
        selectedIdxs = np.random.choice(arr, size=N, replace=False)

        self.X = data_tr.whole.X[selectedIdxs]
        self.Y = np.round(1 - data_tr.whole.Y[selectedIdxs])

        self.whole = dataset_struct(self.X, self.Y)


    def loadRandom(self, N, mean, variance):
        """
        Load random data points sampled from an MVN around training data
        """
        self.X, self.Y = np.ranomd.multivariate_normal(mean, variance, N)


    def getSize(self):
        return len(self.X)

    def get_df(self):
        """
        """
        ## TODO
