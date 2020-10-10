import pandas as pd
import sys
import os
import numpy as np

class datasets():
    def __init__(self, dataPath):
        self.dataPath = dataPath

    def load(self):
        pass


class houseData(datasets):
    def __init__(self):
        self.dataPath = "./datasets/house-processed.csv"

    def load(self):
        df = pd.read_csv(self.dataPath)
        return df
