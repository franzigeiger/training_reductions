import math
from colorsys import hls_to_rgb

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from kymatio.scattering2d.filter_bank import filter_bank
from kymatio.scattering2d.utils import fft2
from skimage.filters import gabor_kernel
from skimage.transform import resize
from torch import nn

from nets import trained_models
from nets.test_models import cornet_s_brainmodel, get_model
from plot.plot_data import plot_images
from utils.correlation import generate_correlation_map, auto_correlation


def visualize_first_layer(model_name, model=None, auto_correlate=False):
    import torch.nn as nn
    if model is None:
        model = cornet_s_brainmodel(model_name, True).activations_model._model
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight
            f_min, f_max = weights.min(), weights.max()
            weights = (weights - f_min) / (f_max - f_min)
            number = math.ceil(math.sqrt(weights.shape[0]))
            filter_weights = weights.data.squeeze()
            img = np.transpose(filter_weights, (0, 2, 3, 1))
            idx = 0
            # fig, axes = pyplot.subplots(ncols=weights.shape[0], figsize=(20, 4))
            for j in range(number):  # in zip(axes, range(weights.shape[0])):
                for i in range(number):
                    ax = plt.subplot(number, number, idx + 1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Kernel {idx}', pad=3)
                    # imgs = img[range(j*8, (j*8)+number)]
                    plt.imshow(img[idx])
                    idx += 1
            plt.show()
            return


def visualize_auto_correlation(model_name, model=None):
    import torch.nn as nn
    if model is None:
        model = get_model(model_name, False, trained_models[model_name])
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.squeeze().numpy()
            number = math.ceil(math.sqrt(weights.shape[0]))
            idx = 0
            for j in range(number):
                for i in range(number):
                    ax = plt.subplot(number, number, idx + 1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'Kernel {idx}', pad=3)
                    img = auto_correlation(weights[idx][0])
                    plt.imshow(img, cmap='gray')
                    idx += 1
            plt.show()
            return


def plot_gabor_filters():
    kernels = []
    labels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1.5, 2):
            for frequency in (0.05, 0.20, 0.30, 0.45):
                for stds in (2, 3):
                    for offset in (0, 1, -1):
                        kernel = np.real(gabor_kernel(frequency, theta=theta,
                                                      sigma_x=sigma, sigma_y=sigma, n_stds=stds, offset=offset))
                        if kernel.shape[0] > 7:
                            overlap = int((kernel.shape[0] - 7) / 2)
                            length = kernel.shape[0]
                            kernel = kernel[overlap:length - overlap, overlap:length - overlap]
                        kernels.append(kernel)
                        labels.append(
                            f'Theta {theta:.2}\n sigma {sigma}, std {stds}\n frequency {frequency}, \noffset {offset}')
                        print(f'Theta {theta}, sigma {sigma}, frequency {frequency}, stds {stds}')
        plot_images(kernels, 10, labels, f'Theta_{theta:.2}')
        kernels = []


def plot_correlation_weights(model_name, size):
    # model = load_model(model_name, False)
    if model_name is 'CORnet-S':
        model_name = 'base'
    model = get_model(model_name, False, trained_models[model_name])
    counter = 0
    # visualize_first_layer('', model)
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d and counter == 0:
            weights = m.weight.data.cpu().numpy()
            length = 64 * size
            matrix = np.empty([length, 0])
            for i in range(0, 64):
                row = np.empty([0, size])
                for j in range(0, 64):
                    if i == j:
                        corr = auto_correlation(weights[i, 0])
                        corr = resize(corr, (size, size),
                                      anti_aliasing=True)
                    else:
                        corr = generate_correlation_map(weights[i, 0], weights[j, 0])
                        corr = resize(corr, (size, size),
                                      anti_aliasing=True)
                    row = np.concatenate((row, corr), axis=0)
                row = (row + 0.25) / (0.5)
                matrix = np.concatenate((matrix, row), axis=1)
                # matrix[0,0] = 1
            # f_min, f_max = np.min(matrix), np.max(matrix)
            # matrix = (matrix - f_min) / (f_max - f_min)
            plot_matrixImage(matrix, model_name + '_correlation_size' + str(size), size)
            return
        elif type(m) == nn.Conv2d:
            counter += 1


def plot_conv2_weights(model_name):
    # model = load_model(model_name, False)
    model = cornet_s_brainmodel(model_name, True).activations_model._model
    counter = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d and counter >= 1:
            weights = m.weight.data.cpu().numpy()
            length = 64 * weights.shape[2]
            matrix = np.empty([length, 0])
            for i in range(0, 64):
                row = np.empty([0, weights.shape[2]])
                for j in range(0, 64):
                    row = np.concatenate((row, weights[i, j]), axis=0)
                f_min, f_max = np.min(row), np.max(row)
                row = (row - f_min) / (f_max - f_min)
                # row[0,0] = 0
                matrix = np.concatenate((matrix, row), axis=1)
                # matrix[0,0] = 1
            # f_min, f_max = np.min(matrix), np.max(matrix)
            # matrix = (matrix - f_min) / (f_max - f_min)
            plot_matrixImage(matrix, model_name + '_conv2')
            return
        elif type(m) == nn.Conv2d:
            counter += 1


def plot_matrixImage(matrix, title, size=3):
    plt.figure(figsize=(20, 20))
    fig, ax = plt.subplots(figsize=(20, 20))
    loc = plticker.MultipleLocator(base=size)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    ax.set_xlabel('Kernels')
    ax.set_ylabel('Filters')
    ax.xaxis.set_ticks(np.arange(0, matrix.shape[0]), size)
    ax.yaxis.set_ticks(np.arange(0, matrix.shape[1]), size)
    ax.grid(which='major', axis='both', linestyle='-')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    extent = (0, matrix.shape[1], matrix.shape[0], 0)
    plt.imshow(matrix, cmap='gray', extent=extent)
    plt.savefig(f'{title}.png')
    plt.show(figsize=(20, 20))


def colorize(z):
    n, m = z.shape
    c = np.zeros((n, m, 3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2 * np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0 / (1.0 + abs(z[idx]) ** 0.3)
    c[idx] = [hls_to_rgb(a, b, 0.8) for a, b in zip(A, B)]
    return c


def normalize(z):
    f_min, f_max = np.min(z), np.max(z)
    z = (z - f_min) / (f_max - f_min)
    return z


def wavelets():
    M = 7
    J = 3
    L = 32
    filters_set = filter_bank(M, M, J, L=L)
    fig, axs = plt.subplots(J, L, sharex=True, sharey=True,
                            gridspec_kw={'width_ratios': [1] * L, 'wspace': 0.5, 'hspace': 0.5, 'top': 0.95,
                                         'bottom': 0.05, 'left': 0.1, 'right': 0.95})
    fig.set_figheight(5)
    fig.set_figwidth(60)
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    i = 0
    func = lambda elems: [(yield f[0][..., 0]) for f in elems]
    plot_wavelets(J, L, filters_set['psi'], func)
    func = lambda elems: [(yield f[0][..., 1]) for f in elems]
    plot_wavelets(J, L, filters_set['psi'], func)
    func = lambda elems: [(yield f[1][..., 0]) for f in elems if len(f) > 3]
    plot_wavelets(J, L, filters_set['psi'], func)
    func = lambda elems: [(yield f[1][..., 1]) for f in elems if len(f) > 3]
    plot_wavelets(J, L, filters_set['psi'], func)
    func = lambda elems: [(yield f[2][..., 0]) for f in elems if len(f) > 4]
    plot_wavelets(J, L, filters_set['psi'], func)
    func = lambda elems: [(yield f[2][..., 1]) for f in elems if len(f) > 4]
    plot_wavelets(J, L, filters_set['psi'], func)


def plot_wavelets(J, L, elems, it):
    fig, axs = plt.subplots(J, L, sharex=True, sharey=True,
                            gridspec_kw={'width_ratios': [1] * L, 'wspace': 0.5, 'hspace': 0.5, 'top': 0.95,
                                         'bottom': 0.05, 'left': 0.1, 'right': 0.95})
    fig.set_figheight(5)
    fig.set_figwidth(60)
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    i = 0
    for f in it(elems):
        filter_c = fft2(f.numpy())
        filter_c = np.fft.fftshift(filter_c)
        if 0 not in filter_c.shape and not np.isnan(filter_c[0]).any():
            axs[i // L, i % L].imshow(normalize(filter_c.astype(float)), cmap='gray')
            # axs[i // L, i % L].imshow(colorize(filter_c))
            axs[i // L, i % L].axis('off')
            axs[i // L, i % L].set_title(
                "j = {} \n theta={}".format(i // L, i % L))
            axs[i // L, i % L].title.set_fontsize(12)
            i = i + 1
    plt.tight_layout()
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    fig.suptitle((r""), fontsize=13)
    fig.show()


if __name__ == '__main__':
    wavelets()
    # plot_gabor_filters()
    # plot_conv2_weights('base')
    # plot_correlation_weights('CORnet-S_train_all_epoch_10', 3)
    # plot_correlation_weights('CORnet-S_train_all_epoch_10', 7)
    # plot_conv2_weights('CORnet-S_train_all_epoch_10')
    # visualize_auto_correlation('CORnet-S_train_gabor_reshape')
    # visualize_first_layer('CORnet-S_train_gabor_reshape_epoch_00')
    # visualize_first_layer('base')
    # plot_correlation_weights('CORnet-S_train_gabor_reshape', 3)
    # plot_correlation_weights('CORnet-S_train_gabor_reshape_epoch_10', 7)
