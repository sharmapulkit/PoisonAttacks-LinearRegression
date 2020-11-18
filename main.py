import numpy as np
import pandas as pd
import pickle as pk

import load_datasets
from attac import *
from protec import *
import models

def test_house_lasso():
    data = load_datasets.houseData()
    data.load()

    poison_data = pk.load(open("poisoned_data", 'rb'))
    data_merged = load_datasets.dataset_struct(pd.concat([data.whole.X, poison_data.X]),
                                        pd.concat([data.whole.Y, poison_data.Y]))

    model = models.Lasso(data.whole.X.shape[1], weight_decay=0.001)
    model.fit(data.train.X, data.train.Y)
    mse_before_poisoning = model.mse(data.train.X, data.train.Y)


    baselinemodel = models.Lasso(data_merged.X.shape[1], weight_decay=0.001)
    # baselinemodel.fit(data_merged.X, data_merged.Y)
    params = pk.load(open("baselineModel_params", 'rb'))
    baselinemodel.setParams(params)

    mse = baselinemodel.mse(data.train.X, data.train.Y)
    print("MSE for lasso, houseData before:", mse_before_poisoning)
    print("MSE for lasso, houseData after poisoning:", mse)

    return mse


def main():
    ### Load Dataset
    # data = load_datasets.houseData()
    data = load_datasets.loanData()
    data.load()

    ### Train Model
    baselinemodel = models.Lasso(data.whole.X.shape[1])
    baselinemodel.fit(data.whole.X, data.whole.Y)
    mse_before_poisoning = baselinemodel.mse(data.train.X, data.train.Y)

    ### Create Attack
    alpha = 0.2 # poisoning ratio
    Num_poisonPts = int(alpha * data.getSize())
    ini_poisonPts = load_datasets.initialDataSet()
    ini_poisonPts.loadInvFlip(data, Num_poisonPts)

    advModel = models.Lasso(data.whole.X.shape[1], weight_decay=0.001)
    bgd = BGD(data, ini_poisonPts, max_iters=30, eta=0.5, line_search_epsilon=0.001, advModel=advModel)
    data_poison = bgd.generatePoisonPoints(baselinemodel)

    pk.dump(data_poison, open("poisoned_data", 'wb'))
    print("Posion points X shape:", data_poison.X.shape)
    print("Posion points Y shape:", data_poison.Y.shape)
    pk.dump(baselinemodel.getParams(), open("baselineModel_params", "wb"))
    print("Train Objective before poisoning:", mse_before_poisoning)

    ### Train with TRIM
    ### Evaluate


if __name__ == "__main__":
    main()
    # test_house_lasso()
