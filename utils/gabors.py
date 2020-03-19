import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from scipy.special import factorial

from plot.plot_data import plot_matrixImage


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


# def tfm_poisson_pdf(x, mu):
#     y, J = transformation_and_jacobian(x)
#     # For numerical stability, compute exp(log(f(x)))
#     return np.exp(y * np.log(mu) - mu - gammaln(y + 1.)) * J


def plot_conv_weights(weights, model_name):
    length = weights.shape[0] * weights.shape[2]
    matrix = np.zeros([length, 0])
    for i in range(0, weights.shape[0]):
        row = np.empty([0, weights.shape[2]])
        for j in range(0, weights.shape[1]):
            row = np.concatenate((row, weights[i, j]), axis=0)
        f_min, f_max = np.min(row), np.max(row)
        row = (row - f_min) / (f_max - f_min)
        # row[0,0] = 0
        matrix = np.concatenate((matrix, row), axis=1)
        # matrix[0,0] = 1
    f_min, f_max = np.min(matrix), np.max(matrix)
    matrix = (matrix - f_min) / (f_max - f_min)
    plot_matrixImage(matrix, 'weights_' + model_name)


def plot_weights(weights, model_name):
    plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(weights.shape[0], weights.shape[1], width_ratios=[1] * weights.shape[1],
                           wspace=0.5, hspace=0.5, top=0.95, bottom=0.05, left=0.1, right=0.95)
    idx = 0
    for i in range(0, weights.shape[0]):
        for j in range(0, weights.shape[1]):
            kernel1 = weights[i, j]
            ax = plt.subplot(gs[i, j])
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(kernel1, cmap='gray')
            ax.set_title(f'Weight component {idx}', pad=2, fontsize=5)
            idx += 1
    plt.tight_layout()
    plt.savefig(f'weights_{model_name}.png')
    plt.show()
    return


def show_kernels(weights, func_name):
    number = math.ceil(math.sqrt(weights.shape[0]))
    img = np.transpose(weights, (0, 2, 3, 1))
    idx = 0
    plt.figure(figsize=(10, 10))
    # fig, axes = pyplot.subplots(ncols=weights.shape[0], figsize=(20, 4))
    for j in range(number):  # in zip(axes, range(weights.shape[0])):
        for i in range(number):
            ax = plt.subplot(number, number, idx + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Kernel {idx}', pad=3)
            # imgs = img[range(j*8, (j*8)+number)]
            channel = img[idx]
            f_min, f_max = channel.min(), channel.max()
            channel = (channel - f_min) / (f_max - f_min)
            plt.imshow(channel)
            idx += 1
    plt.tight_layout()
    plt.savefig(f'kernels_{func_name}.png')
    plt.show()


def similarity(m1, m2):
    sum = 0
    for i in range(m1.shape[0]):
        for j in range(m1.shape[1]):
            sum += np.abs(m1[i, j] - m2[i, j])
    return sum / (m1.shape[0] * m1.shape[1])
