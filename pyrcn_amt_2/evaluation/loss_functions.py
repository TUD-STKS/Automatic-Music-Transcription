from sklearn.metrics import log_loss, mean_squared_error
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr


def mse(y_true, y_pred):
    return mean_squared_error(y_true=y_true, y_pred=y_pred)


def cosine_distance(y_true, y_pred):
    return cosine(u=y_true, v=y_pred)


def correlation(y_true, y_pred):
    return 1 - pearsonr(x=y_true, y=y_pred)[0]


def bce(y_true, y_pred):
    return log_loss(y_true=y_true, y_pred=y_pred)
