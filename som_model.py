import numpy
from copy import deepcopy
from tqdm import tqdm

def winning_neuron(x, W):
    # Also called as Best Matching Neuron/Best Matching Unit (BMU)
    return numpy.argmin(numpy.linalg.norm(x - W, axis=1))


def update_weights(lr, var, x, W, Grid):
    i = winning_neuron(x, W)
    d = numpy.square(numpy.linalg.norm(Grid - Grid[i], axis=1))
    h = numpy.exp(-d/(2 * var * var))
    W = W + lr * h[:, numpy.newaxis] * (x - W)
    return W


def decay_learning_rate(eta_initial, epoch, time_const):
    return eta_initial * numpy.exp(-epoch/time_const)


def decay_variance(sigma_initial, epoch, time_const):
    return sigma_initial * numpy.exp(-epoch/time_const)


def train(X, eta=0.1, epochs=25000, verbose=False):
    som_layer_shape = numpy.array([12, 12])
    X_norm = X / numpy.linalg.norm(X, axis=1).reshape(X.shape[0], 1)
    w = numpy.random.uniform(0, 1, (som_layer_shape[0] * som_layer_shape[1], X.shape[1]))
    w_norm = w / numpy.linalg.norm(w, axis=1).reshape(som_layer_shape[0] * som_layer_shape[1], 1)
    grid = numpy.mgrid[0:som_layer_shape[0], 0:som_layer_shape[1]].reshape(2, som_layer_shape[0] * som_layer_shape[1]).T

    eta_0 = eta
    eta_tau = 1000
    sigma_0 = numpy.max(som_layer_shape) * 0.5
    sigma_tau = 1000/numpy.log10(sigma_0)

    W_new = deepcopy(w_norm)
    eta = deepcopy(eta_0)
    sigma = deepcopy(sigma_0)
    for epoch in range(epochs):
        i = numpy.random.randint(0, X_norm.shape[0])
        W_new = update_weights(eta, sigma, X_norm[i], W_new, grid)
        eta = decay_learning_rate(eta_0, epoch, eta_tau)
        sigma = decay_variance(sigma_0, epoch, sigma_tau)

        if verbose:
          if epoch % 1000 == 0:
            print('Epoch: {}; eta: {}; sigma: {}'.format(epoch, eta, sigma))

    return W_new
