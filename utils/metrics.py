import numpy as np
from sklearn.metrics import r2_score


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((true - pred) / true))


def MSPE(pred, true):
    return np.mean(np.square((true - pred) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    # Flatten arrays for r2_score (requires 2D or 1D input)
    if pred.ndim > 2:
        pred_flat = pred.reshape(pred.shape[0], -1)
        true_flat = true.reshape(true.shape[0], -1)
    else:
        pred_flat = pred
        true_flat = true
    r2 = r2_score(true_flat, pred_flat, multioutput="uniform_average")

    return mae, mse, rmse, mape, mspe, r2
