import math
import random

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from skimage.filters import gabor_kernel
from skimage.transform import resize

from transformations.layer_based import random_state
from utils.correlation import auto_correlation, generate_correlation_map, kernel_convolution, mixture_gaussian
from utils.gabors import gabor_kernel_3


def do_fit_gabor_dist(weights, config):
    params = np.load(config["file"])
    idx = 0
    samples = []
    for i in range(params.shape[2] - 1):
        # if i is 1 or i is 4:
        param = params[:, :, i]
        mu, std = norm.fit(param)
        samples.append(np.random.normal(mu, std, weights.shape[0]))
        # else:
        #     param = params[:,:,i]
        #     result = minimize(negLogLikelihood,  # function to minimize
        #                       x0=np.ones(1),     # start value
        #                       args=(param,),      # additional arguments for function
        #                       method='Powell',   # minimization method, see docs
        #                       )
        #     params.append(np.random.poisson(result.x, weights.shape[0]))

    for k in range(weights.shape[0]):
        for f in range(weights.shape[1]):
            kernel = gabor_kernel_3(samples[0][k], theta=samples[1][k],
                                    sigma_x=samples[2][k], sigma_y=samples[3][k], offset=samples[4][k],
                                    x_c=samples[5][k], y_c=samples[6][k], ks=weights.shape[2])
            if config['reshape']:
                kernel = reshape_with_project(kernel)
            weights[k, f] = kernel
    show_kernels(weights, 'independent_dist')
    return weights


def do_fit_gabor_init(weights, config):
    gabor_params = np.load(config["file"])
    idx = 0
    for kernels in gabor_params:
        for beta in kernels:
            kernel = gabor_kernel_3(beta[0], theta=beta[1],
                                    sigma_x=beta[2], sigma_y=beta[3], offset=beta[4], x_c=beta[5], y_c=beta[6], ks=7)
            if config['reshape']:
                kernel = reshape_with_project(kernel)
            weights[int(idx / 3), idx % 3] = kernel
            idx += 1
    show_kernels(weights, 'fit_gabors')
    return weights


def do_scrumble_gabor_init(weights, config):
    gabor_params = np.load(config["file"])
    idx = 0
    gabor_params = prepare_gabor_params(gabor_params)
    for i in range(gabor_params.shape[1]):
        old_params = gabor_params[:, i]
        randoms = random_state.permutation(gabor_params.shape[0])
        gabor_params[:, i] = gabor_params[randoms, i]

    for filter_params in gabor_params:
        for s, e in ((0, 10), (10, 20), (20, 30)):
            beta = filter_params[s:e]
            filter = gabor_kernel_3(beta[0], theta=np.arctan2(beta[1], beta[2]) / 2,
                                    sigma_x=beta[3], sigma_y=beta[4], offset=np.arctan2(beta[5], beta[6]), x_c=beta[7],
                                    y_c=beta[8],
                                    scale=beta[9], ks=7)
            weights[idx, int(s / 10)] = filter
        idx += 1
    show_kernels(weights, 'scrumble_gabor')
    return weights


def do_gabors(weights, configuration):
    num_theta = weights.shape[0] / 4
    num_frequ = num_theta / 4

    frequency = (0.05, 0.20, 0.30, 0.45)

    # sigma = (1.5, 2)
    # stds = (2, 3)
    # offset = (0, 1, -1)
    choices = [(1.5, 2, 0), (1.5, 2, 1), (1.5, 2, -1), (1.5, 3, 0), (1.5, 3, 1), (1.5, 3, -1),
               (2, 2, 0), (2, 2, 1), (2, 2, -1), (2, 3, 0), (2, 3, 1), (2, 3, -1)]
    idx = 0
    for theta in range(4):
        theta = theta / 4. * np.pi
        for frequency in (0.05, 0.20, 0.30, 0.45):
            configs = random.sample(range(12), int(num_frequ))
            for config in configs:
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=choices[config][0], sigma_y=choices[config][0],
                                              n_stds=choices[config][1], offset=choices[config][2]))
                if kernel.shape[0] > 7:
                    overlap = int((kernel.shape[0] - 7) / 2)
                    length = kernel.shape[0]
                    kernel = kernel[overlap:length - overlap, overlap:length - overlap]
                if configuration['reshape']:
                    kernel = reshape_with_project(kernel)
                weights[idx, 0] = kernel
                weights[idx, 1] = kernel
                weights[idx, 2] = kernel
                idx += 1
    random_order = random_state.permutation(weights.shape[0])
    weights = weights[random_order]
    show_kernels(weights, 'fixed')
    return weights


def show_kernels(weights, func_name):
    f_min, f_max = weights.min(), weights.max()
    weights = (weights - f_min) / (f_max - f_min)
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
            plt.imshow(img[idx])
            idx += 1
    plt.tight_layout()
    plt.savefig(f'kernels_{func_name}.png')
    plt.show()


def do_correlation_init(weights, previous):
    size = weights.shape[2]
    for i in range(0, previous.shape[0]):
        # row = np.empty([0, size])
        for j in range(0, previous.shape[0]):
            if i == j:
                corr = auto_correlation(previous[i, 0])
            else:
                corr = generate_correlation_map(previous[i, 0], previous[j, 0])
            corr1 = resize(corr, (size, size),
                           anti_aliasing=True)
            corr2 = -0.25 + (corr1 + 1) / 4
            weights[i, j] = corr2
    return weights


def do_correlation_init_no_reshape(weights, previous):
    size = weights.shape[2]
    for i in range(0, previous.shape[0]):
        # row = np.empty([0, size])
        for j in range(0, previous.shape[0]):
            if i == j:
                corr = auto_correlation(previous[j].mean(axis=0))
            else:
                corr = generate_correlation_map(previous[j].mean(axis=0), previous[i].mean(axis=0))
            corr1 = resize(corr, (size, size),
                           anti_aliasing=True)
            corr2 = -0.25 + (corr1 + 1) / 4
            weights[i, j] = corr2
    return weights


def reshape_with_project(kernel):
    omin = np.min(kernel)
    omax = np.max(kernel)
    ceiled = (kernel - omin) / (omax - omin)
    kernel = -0.25 + (ceiled * 0.5)
    return kernel


def do_kernel_convolution_init(weights, previous):
    size = weights.shape[2]
    for i in range(0, previous.shape[0]):
        # row = np.empty([0, size])
        for j in range(0, previous.shape[0]):
            corr = kernel_convolution(previous[j].mean(axis=0), previous[i].mean(axis=0))
            weights[i, j] = corr
    return weights


def do_distribution_gabor_init(weights, config):
    params = np.load(f'{config["file"]}')
    param = prepare_gabor_params(params)
    np.random.seed(0)
    samples = mixture_gaussian(param, 64)
    for i in range(weights.shape[0]):
        for s, e in ((0, 10), (10, 20), (20, 30)):
            beta = samples[i, s:e]
            filter = gabor_kernel_3(beta[0], theta=np.arctan2(beta[1], beta[2]) / 2,
                                    sigma_x=beta[3], sigma_y=beta[4], offset=np.arctan2(beta[5], beta[6]), x_c=beta[7],
                                    y_c=beta[8],
                                    scale=beta[9], ks=7)
            weights[i, int(s / 10)] = filter
    show_kernels(weights, 'distribution_init')
    return weights


def prepare_gabor_params(params):
    param = params[:, :, :-1]
    param = param.reshape(64, -1)
    new_param = np.zeros((64, 30))
    for i in range(64):
        for s, e in ((0, 10), (10, 20), (20, 30)):
            p = param[i, int(s * 8 / 10):int(e * 8 / 10)]
            new_param[i, s:e] = (
                p[0], np.sin(2 * p[1]), np.cos(2 * p[1]), p[2], p[3], np.sin(p[4]), np.cos(p[4]), p[5], p[6], p[7])
    return new_param
