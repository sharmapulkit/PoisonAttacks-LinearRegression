import numpy as np
import pandas as pd

import load_datasets
from attac import *
from protec import *
import models

def main():
    ### Load Dataset
    data = load_datasets.houseData()
    data.load()

    ### Train Model
    baselinemodel = models.OLS(data.whole.X.shape[1])
    baselinemodel.fit(data.whole.X, data.whole.Y)

    ### Create Attack
    alpha = 0.2 # poisoning ratio
    Num_poisonPts = int(alpha * data.getSize())
    ini_poisonPts = load_datasets.initialDataSet()
    ini_poisonPts.loadInvFlip(data, Num_poisonPts)

    advModel = models.OLS(data.whole.X.shape[1])
    bgd = BGD(data, ini_poisonPts, max_iters=30, eta=0.1, line_search_epsilon=0.001, advModel=advModel)
    data_poison = bgd.generatePoisonPoints(baselinemodel)

    ### Train with TRIM
    ### Evaluate


if __name__ == "__main__":
    main()
