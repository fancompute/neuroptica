import numpy as np

import neuroptica as neu


def msc_loss(yhat, y):
    return np.sum(1 / 2 * (yhat - y) ** 2)


def d_msc_loss(yhat, y):
    return yhat - y


def one_hot_cross_entropy(y_pred, y_true):
    return np.sum(-1 * y_true * np.log(y_pred))


def d_one_hot_cross_entropy(y_pred, y_true):
    return -1 * y_true / y_pred


if __name__ == "__main__":
    N = 4
    mask = np.ones(N) * 1e-5
    mask[0:2] = 1

    model = neu.Sequential([
        neu.ClementsLayer(N),
        neu.Activation(neu.nonlinearities.Abs(N)),
        neu.ClementsLayer(N),
        neu.Activation(neu.nonlinearities.Abs(N)),
        neu.ClementsLayer(N),
        neu.Activation(neu.nonlinearities.Abs(N)),
        neu.ClementsLayer(N),
        neu.Activation(neu.nonlinearities.Abs(N)),
        neu.ClementsLayer(N),
        neu.Activation(neu.nonlinearities.Abs(N)),
        neu.ClementsLayer(N),
        neu.Activation(neu.nonlinearities.AbsSquared(N)),
        neu.Activation(neu.nonlinearities.Mask(N, mask=mask)),
        neu.Activation(neu.nonlinearities.SoftMax(N))
    ])

    X, Y = neu.utils.generate_ring_planar_dataset()

    P0 = 10
    X_formatted = np.pad(X, (0, N - 2), mode="constant")
    for i, x in enumerate(X_formatted):
        X_formatted[i][2] = np.sqrt(P0 - np.sum(x ** 2))
    Y_formatted = np.pad(Y, (0, N - 2), mode="constant")
    X_formatted = X_formatted.T
    Y_formatted = Y_formatted.T

    # optimizer = neu.InSituGradientDescent(model, one_hot_cross_entropy, d_one_hot_cross_entropy)
    optimizer = neu.InSituGradientDescent(model, msc_loss, d_msc_loss)
    losses = optimizer.fit(X_formatted, Y_formatted, iterations=250, learning_rate=-0.01, batch_size=32)
