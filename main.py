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
    print(data.X.shape)

    ### Train Model
    baselinemodel = models.ols(data.X.shape[1])
    baselinemodel.fit(data)

    ### Create Attack
    bgd = BGD(data, 30)
    data = bgd.generatePoisonPoints(0.1)

    ### Train with TRIM
    ### Evaluate


if __name__ == "__main__":
    main()
