from sklearn.metrics import log_loss, mean_squared_error
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import numpy as np


def mse(y_true, y_pred):
    return mean_squared_error(y_true=y_true, y_pred=y_pred)


def cosine_distance(y_true, y_pred):
    if y_true.ndim > 1:
        loss = []
        for k in range(y_true.shape[1]):
            tmp_loss = cosine(u=y_true[:, k], v=y_pred[:, k])
            if not np.isnan(tmp_loss):
                loss.append(tmp_loss)
        return np.mean(loss)
    else:
        return cosine(u=y_true, v=y_pred)


def correlation(y_true, y_pred):
    return 1 - pearsonr(x=y_true.flatten(), y=y_pred.flatten())[0]


def bce(y_true, y_pred):
    return log_loss(y_true=y_true, y_pred=y_pred)
