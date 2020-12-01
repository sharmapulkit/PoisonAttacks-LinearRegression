#!/bin/sh
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
    data = load_datasets.houseData()
    # data = load_datasets.loanData()
    data.load()

    print("Attack on : Ridge; RVO")
    ### Train Model
    baselinemodel = models.Ridge(data.whole.X.shape[1])
    baselinemodel.fit(data.whole.X, data.whole.Y, max_iter=400)
    mse_before_poisoning = baselinemodel.mse(data.train.X, data.train.Y)

    ### Create Attack
    # poisoning_rate = 0.08
    mses = {}
    # for poisoning_rate in [0, 0.04, 0.08, 0.12, 0.16, 0.20]:
    for poisoning_rate in [0.20]:
        alpha = poisoning_rate / (1 - poisoning_rate) # poisoning rate
        Num_poisonPts = int(alpha * data.getSize())
        ini_poisonPts = load_datasets.initialDataSet()
        ini_poisonPts.loadInvFlip(data, Num_poisonPts)

        advModel = models.Ridge(data.whole.X.shape[1])
        bgd = BGD(data, ini_poisonPts, max_iters=50, eta=0.01, line_search_epsilon=1e-8, advModel=advModel, rvo=True)
        data_poison, mse_after_poisoning = bgd.generatePoisonPoints(baselinemodel)
        mse_val_after_poisoing = bgd.advModel.objective(data.val.X, data.val.Y)

        mses[poisoning_rate] = mse_val_after_poisoning
        pk.dump(baselinemodel.getParams(), open("baselineModel_params_ridge_{}_bgd.params".format(poisoning_rate), "wb"))
        print("Posion points X shape:", data_poison.X.shape)
        print("Posion points Y shape:", data_poison.Y.shape)
        print("Final Val mse:", mse_val_after_poisoning)
        print("Final train mse:", bgd.advModel.objective(data.train.X, data.train.Y))

    print(mses)
    pk.dump(mses, open("poisoned_ridge_mses_rvo", 'wb'))
    print("Train Objective before poisoning:", mse_before_poisoning)

    ### Train with TRIM
    ### Evaluate


if __name__ == "__main__":
    main()
    # test_house_lasso()
