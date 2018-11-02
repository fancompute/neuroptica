import matplotlib.pyplot as plt
import numpy as np

import neuroptica as neu

if __name__ == "__main__":

    # X, Y = neu.utils.generate_ring_planar_dataset()
    # X, Y = neu.utils.generate_diagonal_planar_dataset()
    X, Y = neu.utils.generate_separable_planar_dataset()

    labels = np.array([0 if yi[0] > yi[1] else 1 for yi in Y]).flatten()
    plt.figure(figsize=(6, 6))
    plt.scatter((X.T)[0, :], (X.T)[1, :], c=labels, cmap=plt.cm.Spectral)
    plt.colorbar()
    plt.show()


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

    P0 = 10
    X_formatted = np.array(np.pad(X, (0, N - 2), mode="constant"))
    for i, x in enumerate(X_formatted):
        X_formatted[i][2] = np.sqrt(P0 - np.sum(x ** 2))
    Y_formatted = np.pad(Y, (0, N - 2), mode="constant")
    X_formatted = X_formatted.T
    Y_formatted = Y_formatted.T

    # optimizer = neu.InSituGradientDescent(model, one_hot_cross_entropy, d_one_hot_cross_entropy)
    optimizer = neu.InSituGradientDescent(model, neu.losses.MeanSquaredError)
    losses = optimizer.fit(X_formatted, Y_formatted, iterations=2000, learning_rate=0.001, batch_size=32)
