import numpy as np
from scipy.special import factorial


def normalize(X):
    f_min, f_max = X.min(), X.max()
    return (X - f_min) / (f_max - f_min)


def gabor_kernel_2(frequency, sigma_x, sigma_y, theta=0, offset=0, ks=61):
    w = np.floor(ks / 2)
    y, x = np.mgrid[-w:w + 1, -w:w + 1]
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)
    g = np.zeros(y.shape)
    g[:] = np.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2))
    g /= 2 * np.pi * sigma_x * sigma_y
    g *= np.cos(2 * np.pi * frequency * rotx + offset)
    return g


def gabor_kernel_3(frequency, x_c, y_c, sigma_x, sigma_y, theta=0, offset=0, ks=61, scale=1):
    w = np.floor(ks / 2)
    y, x = np.mgrid[-w:w + 1, -w:w + 1]
    rotx = (x - x_c) * np.cos(theta) + (y - y_c) * np.sin(theta)
    roty = -(x - x_c) * np.sin(theta) + (y - y_c) * np.cos(theta)
    g = np.zeros(y.shape)
    g[:] = np.exp(-0.5 * (rotx ** 2 / sigma_x ** 2 + roty ** 2 / sigma_y ** 2))
    g /= 2 * np.pi * sigma_x * sigma_y
    g *= np.cos(2 * np.pi * frequency * rotx + offset)
    return g * scale


def poisson(k, lamb):
    """poisson pdf, parameter lamb is the fit parameter"""
    return (lamb ** k / factorial(k)) * np.exp(-lamb)


def negLogLikelihood(params, data):
    """ the negative log-Likelohood-Function"""
    lnl = - np.sum(np.log(poisson(data, params[0])))
    return lnl


def tfm_poisson_pdf(x, mu):
    y, J = transformation_and_jacobian(x)
    # For numerical stability, compute exp(log(f(x)))
    return np.exp(y * np.log(mu) - mu - gammaln(y + 1.)) * J
