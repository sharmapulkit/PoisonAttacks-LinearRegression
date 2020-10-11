import models
import load_datasets

class attack():
    def __init__(self):
        pass

class BGD(attack):
    """
    Perform a BGD attack on a given model
    """
    def __init__(self, data, max_iters):
        super().__init__()
        self._max_iters = max_iters

    def _generatePoisonPoints(self, model, advmodel, data_tr, data_val, ini_poisonPts, epsilon):
        i = 0
        dataUnion = pd.concat([data_tr.X, ini_poisonPts.X])
        model.fit(dataUnion)
        theta = model.getParams()
        wPrev = 0
        wCurr = 0
        while (i < self._max_iters):
            if (i != 0 and np.abs(wCurr - wPrev) > epsilon):
                break

            wCurr = advmodel.objective(data_val)
            dataUnion = data_tr
            for c in range(1, len(ini_poisonPts)+1):
                x = advmodel.findDataPoint(data_val)
                dataUnion = pd.concat([dataUnion, x])
                model.fit(dataUnion)
                theta = model.getParams()
                wPrev = wCurr
                wCurr = advmodel.objective(data_val)
            i += 1
        poisonPoints = dataUnion[-len(ini_poisonPts):]
        return poisonPoints

    def set_advmodel(self, m):
        self.advmodel = m

    def set_model(self, m):
        self.model = m

    def generatePoisonPoints(self, alpha):
        """ Return a set of poison points """
        epsilon = 0.01
        trainData = load_datasets.houseData()
        trainData.load()

        Num_poisonPts = int(alpha*trainData.getSize())
        ini_poisonPts = load_datasets.initialDataSet()
        ini_poisonPts.loadInvFlip(trainData, Num_poisonPts)
        print(ini_poisonPts.X.shape)

        #poisonPoints = self._generatePoisonPoints(trainData, ini_poisonPts, epsilon)

        #return poisonPoints



