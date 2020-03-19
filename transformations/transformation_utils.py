import random

import cornet
import numpy as np
import scipy.stats as st
import torch
from scipy.stats import norm
from skimage.filters import gabor_kernel
from skimage.transform import resize

from transformations.layer_based import random_state
from utils.correlation import auto_correlation, generate_correlation_map, kernel_convolution
from utils.distributions import mixture_gaussian, best_fit_distribution
from utils.gabors import gabor_kernel_3, plot_conv_weights, show_kernels

layers = ['V1.conv1', 'V1.conv2',
          'V2.conv_input', 'V2.skip', 'V2.conv1', 'V2.conv2', 'V2.conv3',
          'V4.conv_input', 'V4.skip', 'V4.conv1', 'V4.conv2', 'V4.conv3',
          'IT.conv_input', 'IT.skip', 'IT.conv1', 'IT.conv2', 'IT.conv3']


def do_fit_gabor_dist(weights, config, **kwargs):
    # fit independent gaussians for each gabor parameter
    params = np.load(config["file"])
    idx = 0
    samples = []
    for i in range(params.shape[2] - 1):
        param = params[:, :, i]
        mu, std = norm.fit(param)
        samples.append(np.random.normal(mu, std, weights.shape[0]))
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


def do_fit_gabor_init(weights, config, **kwargs):
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


def do_kernel_normal_distribution_init(weights, **kwargs):
    for i in range(weights.shape[0]):
        flat = weights[i].flatten()
        mu, std = norm.fit(flat)
        weights[i] = np.random.normal(mu, std, weights[i].shape)
    return weights


def do_layer_normal_distribution_init(weights, **kwargs):
    flat = weights.flatten()
    mu, std = norm.fit(flat)
    return np.random.normal(mu, std, weights.shape)


def do_scrumble_gabor_init(weights, config, **kwargs):
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
    show_kernels(weights, config, 'scrumble_gabor')
    return weights


def do_gabors(weights, config, **kwargs):
    num_theta = weights.shape[0] / 4
    num_frequ = num_theta / 4
    configuration = config
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


def do_correlation_init(weights, previous, **kwargs):
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


def do_correlation_init_no_reshape(weights, previous, **kwargs):
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


def do_kernel_convolution_init(weights, previous, **kwargs):
    size = weights.shape[2]
    for i in range(0, previous.shape[0]):
        # row = np.empty([0, size])
        for j in range(0, previous.shape[0]):
            corr = kernel_convolution(previous[j].mean(axis=0), previous[i].mean(axis=0))
            weights[i, j] = corr
    return weights


def do_distribution_gabor_init(weights, config, index, **kwargs):
    if index != 0:
        params = np.load(config[f'file_{index}'])
    else:
        params = np.load(f'{config["file"]}')
    param, tuples = prepare_gabor_params(params)
    np.random.seed(0)
    components = config[f'comp_{index}'] if f'comp_{index}' in config else 0
    samples = mixture_gaussian(param, weights.shape[0], components, f'gabor_{index}')
    for i in range(weights.shape[0]):
        for s, e in tuples:
            beta = samples[i, s:e]
            filter = gabor_kernel_3(beta[0], theta=np.arctan2(beta[1], beta[2]) / 2,
                                    sigma_x=beta[3], sigma_y=beta[4], offset=np.arctan2(beta[5], beta[6]), x_c=beta[7],
                                    y_c=beta[8],
                                    scale=beta[9], ks=weights.shape[-1])
            weights[i, int(s / 10)] = filter
    if weights.shape[1] == 3:
        show_kernels(weights, 'distribution_init')
    else:
        plot_conv_weights(weights, 'distribution_init_kernel')
    return weights


def do_distribution_gabor_init_channel(weights, config, index, **kwargs):
    if index != 0:
        params = np.load(config[f'file_{index}'])
    else:
        params = np.load(f'{config["file"]}')
    param, tuples = prepare_gabor_params_channel(params)
    np.random.seed(0)
    components = config[f'comp_{index}'] if f'comp_{index}' in config else 0
    samples = mixture_gaussian(param, weights.shape[0], components)
    for i in range(weights.shape[0]):
        beta = samples[i]
        filter = gabor_kernel_3(beta[0], theta=np.arctan2(beta[1], beta[2]) / 2,
                                sigma_x=beta[3], sigma_y=beta[4], offset=np.arctan2(beta[5], beta[6]), x_c=beta[7],
                                y_c=beta[8],
                                scale=beta[9], ks=weights.shape[-1])
        weights[int(i / weights.shape[0]), int(i % weights.shape[0])] = filter
    if weights.shape[1] == 3:
        show_kernels(weights, 'distribution_init_channel_color')
    else:
        plot_conv_weights(weights, 'distribution_init_channel')
    return weights


def do_distribution_weight_init(weights, config, index, **kwargs):
    # dimension = 0, kernel level, dimension 1 channel level
    # assume weights are untrained weights
    trained = cornet.cornet_s(pretrained=True, map_location=torch.device('cpu'))
    for sub in ['module'] + layers[index].split('.'):
        trained = trained._modules.get(sub)
    weights = trained.weight.data.cpu().numpy()
    dim = config['dim'] if 'dim' in config else config[f'dim_{index}']
    components = config[f'comp_{index}'] if f'comp_{index}' in config else 0
    if dim == 0:
        params = weights.reshape(weights.shape[0], weights.shape[1], -1)
        params = params.reshape(params.shape[0], -1)
    else:
        params = weights.reshape(-1, weights.shape[2], weights.shape[3])
        params = params.reshape(params.shape[0], -1)

    np.random.seed(0)
    samples = mixture_gaussian(params, params.shape[0], components, f'weight_dim{dim}_{index}')
    idx = 0

    if dim is 0:
        new_weights = samples.reshape(weights.shape[0], weights.shape[1], weights.shape[2] * weights.shape[3])
        new_weights = new_weights.reshape(weights.shape[0], weights.shape[1], weights.shape[2], weights.shape[3])
        weights = new_weights
    else:
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                weights[i, j] = samples[idx].reshape(weights.shape[2], weights.shape[3])
                idx += 1
    # show_kernels(weights, 'distribution_init')
    return weights


def prepare_gabor_params(params, **kwargs):
    tuples = []
    for i in range(params.shape[1]):
        tuples.append((i * 10, (i + 1) * 10))
    param = params[:, :, :-1]
    param = param.reshape(params.shape[0], -1)
    new_param = np.zeros((params.shape[0], params.shape[1] * 10))
    for i in range(64):
        for s, e in tuples:
            p = param[i, int(s * 8 / 10):int(e * 8 / 10)]
            new_param[i, s:e] = (
                p[0], np.sin(2 * p[1]), np.cos(2 * p[1]), p[2], p[3], np.sin(p[4]), np.cos(p[4]), p[5], p[6], p[7])
    return new_param, tuples


def prepare_gabor_params_channel(params, **kwargs):
    tuples = [(0, 10)]
    param = params[:, :, :-1]
    param = param.reshape(-1, 8)
    new_param = np.zeros((param.shape[0], 10))
    for i in range(param.shape[0]):
        p = param[i]
        new_param[i] = (
            p[0], np.sin(2 * p[1]), np.cos(2 * p[1]), p[2], p[3], np.sin(p[4]), np.cos(p[4]), p[5], p[6], p[7])
    return new_param, tuples


def do_best_dist_init_layer(weights, **kwargs):
    name, params = best_fit_distribution(weights)
    print(f'Best fit distribution: {name}, params: {params}')
    best_dist = getattr(st, name)
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    return best_dist.rvs(size=weights.shape, *arg, loc=loc, scale=scale)


def do_best_dist_init_kernel(weights, **kwargs):
    dists = {}
    for i in weights:
        name, params = best_fit_distribution(i)
        if not name in dists:
            dists[name] = 1
        else:
            dists[name] += 1
        print(f'Best fit distribution for kernel: {name}')

    max = 0
    best = ''
    for k, v in dists.items():
        if v > max:
            best = k
    best_dist = getattr(st, best)
    for i in range(weights.shape[0]):
        p = best_dist.fit(weights[i])
        print(f'Best fit distribution: {best} with params {p}')
        arg = p[:-2]
        loc = p[-2]
        scale = p[-1]
        # print(weights[i])
        weights[i] = best_dist.rvs(size=weights[i].shape, *arg, loc=loc, scale=scale)
        # print(weights[i])

    return weights


def do_in_channel_jumbler(weights, **kwargs):
    if len(weights.shape) > 2:
        for i in range(weights.shape[1]):
            random_order = random_state.permutation(weights.shape[0])
            weights[:, i] = weights[random_order, i]
    return weights
