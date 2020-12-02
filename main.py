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
    # data = load_datasets.houseData(train_frac=0.22)
    data = load_datasets.pharmpreprocData(train_frac=0.14)
    # data = load_datasets.loanData(train_frac=0.006)
    data.load()

    model_name = "Lasso"
    strategy = "RVO"
    datasetName = "Health"
    print("Attack on InvFlip: {}; {}".format(model_name, strategy))
    ### Train Model

    ### Create Attack
    # poisoning_rate = 0.08
    mses = {}
    for poisoning_rate in [0.2]:
    # for poisoning_rate in [0, 0.04, 0.08, 0.12, 0.16, 0.20]:
        baselinemodel = models.Ridge(data.train.X.shape[1])
        baselinemodel.fit(data.train.X, data.train.Y, max_iter=400)
        mse_before_poisoning = baselinemodel.mse(data.train.X, data.train.Y)

        alpha = poisoning_rate / (1 - poisoning_rate) # poisoning rate
        Num_poisonPts = int(alpha * data.getTrain().getSize() )
        ini_poisonPts = load_datasets.initialDataSet()
        ini_poisonPts.loadInvFlip(data, Num_poisonPts)
        # ini_poisonPts.loadBFlip(data, Num_poisonPts)

        advModel = models.Ridge(data.whole.X.shape[1])
        bgd = BGD(data, ini_poisonPts, max_iters=50, eta=0.02, line_search_epsilon=1e-8, advModel=advModel, rvo=True)
        data_poison, mse_after_poisoning = bgd.generatePoisonPoints(baselinemodel)
        mse_val_after_poisoning = bgd.advModel.objective(data.val.X, data.val.Y)

        mses[poisoning_rate] = mse_val_after_poisoning
        # pk.dump(baselinemodel.getParams(), open("baselineModel_params_{}_{}_{}_{}.params".format(model_name, poisoning_rate, strategy, datasetName), "wb"))
        print("Posion points X shape:", data_poison.X.shape)
        print("Posion points Y shape:", data_poison.Y.shape)
        print("Final Val mse:", mse_val_after_poisoning)
        print("Final train mse:", bgd.advModel.objective(data.train.X, data.train.Y))
        poisoned_data = np.concatenate((np.concatenate((data.train.Y, data_poison.Y)), np.concatenate((data.train.X, data_poison.X))), axis=1)
        # pk.dump(poisoned_data, open("poisoned_dataset/poisoned_data_{}_{}_{}_p{}".format(model_name, strategy, datasetName, poisoning_rate), 'wb'))
        # print("Saved poisoned dataset", poisoned_data.shape)

    print("MSE:", mses)    
    pk.dump(mses, open("poisoned_{}_mses_{}_data_{}_p0.12".format(model_name, strategy, datasetName), 'wb'))
    print("Train Objective before poisoning:", mse_before_poisoning)


if __name__ == "__main__":
    main()
    # test_house_lasso()
