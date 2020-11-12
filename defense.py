import numpy as np
import scipy.optimize
from models import Model

def trim_defense(X, y, n, model: Model, n_iter=400, epsilon=1e-5):
    """
       Trains the model using TRIM algorithm 
    """
    N, _ = X.shape
    train_indices = np.random.permutation(N)[:n]
    x_train = X[train_indices]
    y_train = y[train_indices]
    model.fit(x_train, y_train)
    curr_obj = model.objective(x_train, y_train)
    for i in range(n_iter):
        curr_w = model.w
        curr_b = model.b
        loss_i = np.square(X.dot(curr_w) + curr_b - y)
        sort_indices = np.argsort(loss_i)[:n]
        x_train = X[sort_indices]
        y_train = y[sort_indices]
        model.fit(x_train, y_train)
        new_obj = model.objective(x_train, y_train)
        if curr_obj - new_obj > epsilon:
            break

        curr_obj = new_obj
