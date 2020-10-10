import pandas as pd
import sys
import os
import numpy as np

class datasets():
    def __init__(self):
        pass

    def load(self):
        pass


class houseData(datasets):
    def __init__(self):
        self.dataPath = "./datasets/house-processed.csv"

    def load(self):
        self.df = pd.read_csv(self.dataPath)
        target_columns = ['SalePrice']
        feature_columns = self.df.columns.drop(target_columns)

        self.X = self.df[feature_columns]
        self.Y = self.df[target_columns]

    def get_df(self):
        return self.df

class initialDataSet(datasets):
    def __init__(self):
        pass

    def loadInvFlip(self, data_tr, N):
        """
        Load an Inverse Flip data set randomly from training data
        """
        trainDataSize = len(data_tr)
        assert N < trainDataSize

        arr = range(0, trainDataSize)
        selectedIdxs = np.random.choice(arr, size=N, replace=False)

        self.X = data_tr.X[selectedIdxs]
        self.Y = 1 - data_tr.Y[selectedIdxs]


    def loadBFlip(self, data_tr, N):
        """
        Load a Boundary Flip data set randomly from training data
        """
        trainDataSize = len(data_tr)
        assert N < trainDataSize

        arr = range(0, trainDataSize)
        selectedIdxs = np.random.choice(arr, size=N, replace=False)

        self.X = data_tr.X[selectedIdxs]
        self.Y = round(1 - data_tr.Y[selectedIdxs])


    def loadRandom(self, N, mean, variance):
        """
        Load random data points sampled from an MVN around training data
        """
        self.X, self.Y = np.ranomd.multivariate_normal(mean, variance, N)


    def get_df(self):
        """
        """
        ## TODO
