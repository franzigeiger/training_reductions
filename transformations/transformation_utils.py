import pickle
import random
from os import path

import numpy as np
import scipy.stats as st
import torch
from scipy.stats import norm
from skimage.filters import gabor_kernel
from skimage.transform import resize
from sklearn import feature_selection

from nets import global_data
from transformations.layer_based import random_state
from utils.correlation import auto_correlation, generate_correlation_map, kernel_convolution
from utils.distributions import mixture_gaussian, best_fit_distribution, poisson_sample
from utils.gabors import gabor_kernel_3, show_kernels


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


def do_kernel_normal_distribution_init(weights, shape, **kwargs):
    old_kernels = weights.shape[0]
    new_kernels = shape[0]
    new_weights = np.zeros(shape)
    if old_kernels < new_kernels:
        rand = random.choices(range(old_kernels), k=new_kernels - old_kernels)
        base = list(range(old_kernels))
        indexes = base + rand
    else:
        indexes = list(range(new_kernels))
    for i in range(new_kernels):
        flat = weights[indexes[i]].flatten()
        mu, std = norm.fit(flat)
        new_weights[i] = np.random.normal(mu, std, shape[1:])
    # print('Corrected kernel norm version!')
    return new_weights


# diesn't work
def do_channel_normal_distribution_init(weights, shape, **kwargs):
    old_kernels = weights.shape[0]
    new_kernels = weights.shape[0]
    old_channels = weights.shape[1]
    new_channels = weights.shape[1]
    new_weights = np.zeros(shape)
    if old_kernels < new_kernels:
        rand = np.random.random((new_kernels, old_kernels)) * old_kernels
        base = list(range(old_kernels))
        indexes = base + int(rand)
    else:
        indexes = list(range(new_kernels))
    if old_channels < new_channels:
        rand = np.random.random((new_channels, old_channels)) * old_channels
        base = list(range(old_channels))
        indexes_channel = base + int(rand)
    else:
        indexes_channel = list(range(new_kernels))

    for i in range(new_kernels):
        for j in range(new_channels):
            flat = weights[indexes[i], indexes_channel[j]].flatten()
            mu, std = norm.fit(flat)
            new_weights[i, j] = np.random.normal(mu, std, shape[2:])
    print('Corrected channel norm version!')
    return new_weights


def do_layer_normal_distribution_init(weights, shape, **kwargs):
    flat = weights.flatten()
    mu, std = norm.fit(flat)
    return np.random.normal(mu, std, shape)


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


def do_distribution_gabor_init(weights, config, index, shape, **kwargs):
    if index != 0:
        params = np.load(config[f'file_{index}'])
    else:
        params = np.load(f'{config["file"]}')
    param, tuples = prepare_gabor_params(params)
    # np.random.seed(0)
    components = config[f'comp_{index}'] if f'comp_{index}' in config else 0
    samples = mixture_gaussian(param, shape[0], components, f'gabor_{index}')
    new_weights = np.zeros(shape)
    for i in range(shape[0]):
        for s, e in tuples:
            beta = samples[i, s:e]
            filter = gabor_kernel_3(beta[0], theta=np.arctan2(beta[1], beta[2]) / 2,
                                    sigma_x=beta[3], sigma_y=beta[4], offset=np.arctan2(beta[5], beta[6]), x_c=beta[7],
                                    y_c=beta[8],
                                    scale=beta[9], ks=shape[-1])
            new_weights[i, int(s / 10)] = filter
    # if weights.shape[1] == 3:
    #     show_kernels(weights, 'distribution_init')
    # else:
    #     plot_conv_weights(weights, 'distribution_init_kernel')
    return new_weights


def do_distribution_gabor_init_channel(weights, config, index, shape, **kwargs):
    if index != 0:
        params = np.load(config[f'file_{index}'])
    else:
        params = np.load(f'{config["file"]}')
    param, tuples = prepare_gabor_params_channel(params)
    # np.random.seed(0)
    components = config[f'comp_{index}'] if f'comp_{index}' in config else 0
    samples = mixture_gaussian(param, shape[0], components)
    for i in range(shape[0]):
        beta = samples[i]
        filter = gabor_kernel_3(beta[0], theta=np.arctan2(beta[1], beta[2]) / 2,
                                sigma_x=beta[3], sigma_y=beta[4], offset=np.arctan2(beta[5], beta[6]), x_c=beta[7],
                                y_c=beta[8],
                                scale=beta[9], ks=shape[-1])
        weights[int(i / shape[0]), int(i % shape[0])] = filter
    # if weights.shape[1] == 3:
    #     show_kernels(weights, 'distribution_init_channel_color')
    # else:
    #     plot_conv_weights(weights, 'distribution_init_channel')
    return weights


def do_distribution_weight_init(weights, config, index, shape, **kwargs):
    # dimension = 0, kernel level, dimension 1 channel level
    # assume weights are untrained weights
    # trained = cornet.cornet_s(pretrained=True, map_location=torch.device('cpu'))
    # for sub in ['module'] + layers[index].split('.'):
    #     trained = trained._modules.get(sub)
    # weights = trained.weight.data.cpu().numpy()
    dim = config['dim'] if 'dim' in config else config[f'dim_{index}']
    components = config[f'comp_{index}'] if f'comp_{index}' in config else 0
    if dim == 0:
        params = weights.reshape(weights.shape[0], weights.shape[1], -1)
        params = params.reshape(params.shape[0], -1)
        samples = shape[0]
    else:
        params = weights.reshape(-1, weights.shape[2], weights.shape[3])
        params = params.reshape(params.shape[0], -1)
        samples = shape[0] * shape[1]

    # np.random.seed(0)
    samples = mixture_gaussian(params, samples, components, f'weight_dim{dim}_{index}')
    idx = 0
    new_weights = np.zeros(shape)
    if dim is 0:
        new_weights = samples.reshape(shape[0], shape[1], shape[2] * shape[3])
        new_weights = new_weights.reshape(shape[0], shape[1], shape[2], shape[3])
    else:
        for i in range(shape[0]):
            for j in range(shape[1]):
                new_weights[i, j] = samples[idx].reshape(shape[2], shape[3])
                idx += 1
    # if weights.shape[1] == 3:
    #     show_kernels(weights, 'distribution_init_channel_color')
    # else:
    #     plot_conv_weights(weights, 'distribution_init_channel')
    return new_weights


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


def do_best_dist_init_kernel(weights, shape, index, **kwargs):
    name = f'best_dist_kernel_{index}'
    dir = '/braintree/home/fgeiger/weight_initialization/'
    print(f'Best dist with name {name}')
    if path.exists(f'{dir}/{name}.pkl'):
        samples, best = load_distribution(name, dir)
        if weights.shape == shape and global_data.seed == 0:
            print(f'We use stored weights for best distribution {best}')
            return samples
    else:
        dists = {}
        index = 0
        for i in weights:
            print(f'Fit for kernel {index}')
            index += 1
            dist, params = best_fit_distribution(i)
            if not dist in dists:
                dists[dist] = 1
            else:
                dists[dist] += 1
            # print(f'Best fit distribution for kernel: {name}')

        max = 0
        best = ''
        for k, v in dists.items():
            if v > max:
                best = k
    best_dist = getattr(st, best)
    old_shape = weights.shape[0]
    new_weights = np.zeros(shape)
    for i in range(shape[0]):
        p = best_dist.fit(weights[i % old_shape])
        print(f'Best fit distribution: {best} with params {p}')
        arg = p[:-2]
        loc = p[-2]
        scale = p[-1]
        new_weights[i] = best_dist.rvs(size=shape[1:], *arg, loc=loc, scale=scale)
    if shape == weights.shape and global_data.seed == 0:
        save_distribution(name, dir, new_weights, best)
    return new_weights


def save_distribution(name, dir, samples, distribution):
    print(f'Save best distribution {distribution} for layer: {name}')
    dict = {'samples': samples, 'distribution': distribution}
    pickle_out = open(f'{dir}/{name}.pkl', "wb")
    pickle.dump(dict, pickle_out)


def load_distribution(name, dir):
    pickle_in = open(f'{dir}/{name}.pkl', "rb")
    dist = pickle.load(pickle_in)
    return dist['samples'], dist['distribution']


def do_in_channel_jumbler(weights, **kwargs):
    if len(weights.shape) > 2:
        for i in range(weights.shape[1]):
            random_order = random_state.permutation(weights.shape[0])
            weights[:, i] = weights[random_order, i]
    return weights


def do_in_kernel_jumbler(weights, **kwargs):
    if len(weights.shape) > 2:
        for i in range(weights.shape[0]):
            random_order = random_state.permutation(weights.shape[1])
            weights[i] = weights[i, random_order]
    return weights


def do_dist_in_layer(weights, config, shape, **kwargs):
    name = config['distribution']
    distribution = getattr(st, name)
    params = distribution.fit(weights)
    print(f'Use distribution: {name}, params: {params}')
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    return distribution.rvs(size=shape, *arg, loc=loc, scale=scale)


def do_dist_in_kernel(weights, config, shape, **kwargs):
    name = config['distribution']
    best_dist = getattr(st, name)
    print(f'Use distribution: {name}')
    old_shape = weights.shape[0]
    new_weights = np.zeros(shape)
    for i in range(shape[0]):
        p = best_dist.fit(weights[i % old_shape])
        arg = p[:-2]
        loc = p[-2]
        scale = p[-1]
        new_weights[i] = best_dist.rvs(size=shape[1:], *arg, loc=loc, scale=scale)
    return new_weights


def do_poisson_layer(weights, shape, **kwargs):
    return poisson_sample(weights, shape)


def do_poisson_kernel(weights, shape, **kwargs):
    old_shape = weights.shape[0]
    new_weights = np.zeros(shape)
    for i in range(shape[0]):
        new_weights[i] = poisson_sample(weights[i % old_shape], shape[1:])
    return new_weights


def prev_std_init(weights, previous, config, index, **kwargs):
    params = config[f'params_{index}']
    p = np.poly1d(params)
    for i in range(previous.shape[0]):
        prev = np.std(previous[i])
        res = p(prev)
        rand = np.random.normal(0, res, weights.shape[0])
        rand.resize([weights.shape[0], 1, 1])
        weights[:, i] = rand
    return weights


def prev_std_init_single(weights, previous, config, index, shape, **kwargs):
    # params = config[f'params_{index}']
    # p = np.poly1d(params)
    new_weights = np.zeros(shape)
    for i in range(previous.shape[0]):
        mu, std = norm.fit(previous[i])
        # res = p(prev)
        rand = np.random.normal(mu, std, shape[0] * shape[2] * shape[3])
        rand.resize([shape[0], shape[2], shape[3]])
        new_weights[:, i] = rand
    return new_weights


def do_sign_init_layer(weights, previous, config, index, **kwargs):
    distr = config[f'params_{index}']
    std = np.std(previous.squeeze())
    return np.random.choice([-std, std, 0], weights.shape, p=distr)


def do_sign_init_kernel(weights, previous, config, index):
    distr = config[f'params_{index}']
    for i in range(previous.shape[0]):
        std = np.std(previous[i].squeeze())
        rand = np.random.choice([-std, std, 0], weights.shape[0], p=distr)
        rand.resize([weights.shape[0], 1, 1])
        weights[:, i] = rand
    return weights


def do_mutual_information(weights, previous, config, **kwargs):
    assert weights.shape[0] == weights.shape[1]
    kernels = previous.shape[0]
    for i in range(kernels):
        for j in range(kernels):
            # print(f'Score kernel {i} and {j}')
            weights[i, j] = feature_selection.mutual_info_regression(previous[i].flatten().reshape(-1, 1),
                                                                     previous[j].flatten())
    print(f'Kernel mean mutual information {np.mean(weights)}')
    return weights


def do_eigenvalue_channel(weights, shape, previous, config, **kwargs):
    new_weights = np.zeros(shape)
    kernels = weights.shape[0]
    for i in range(kernels):
        for j in range(weights.shape[1]):
            # print(f'Score kernel {i} and {j}')
            values = np.linalg.eigvals(weights[i, j])
            new_weights[i, j] = np.identity(shape[-1]) * values[0]
    return new_weights


def do_eigenvalue_channel_dist(weights, shape, previous, config, **kwargs):
    new_weights = np.zeros(shape)
    kernels = weights.shape[0]
    channels = weights.shape[1]
    eigenvalues = []
    for i in range(kernels):
        for j in range(channels):
            # print(f'Score kernel {i} and {j}')
            values = np.linalg.eigvals(weights[i, j])
            eigenvalues.append(values[0])
    mean, std = norm.fit(eigenvalues)
    for i in range(kernels):
        for j in range(channels):
            new_weights[i, j] = np.identity(shape[-1]) * np.random.normal(mean, std)
    return new_weights


# dataloader = LazyLoad(lambda : get_dataloader(image_load=2000, batch_size=2000, workers=1))
# dataloader = get_dataloader(image_load=2000, batch_size=2000, workers=1)

def do_batch_from_image_init(module, previous, model, previous_module, previous_name, **kwargs):
    for (number, inp) in enumerate([]):
        activation = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()

            return hook

        previous_module.register_forward_hook(get_activation(previous_name))
        model(inp[0])
        print(activation)
        # normalize based on batch:
        activ = activation[previous_name]
        beta = activ.mean([0, 2, 3])
        gamma = torch.sqrt(activ.var([0, 2, 3], unbiased=False))
        # activ = activ - activ.mean([0, 2, 3])[None, :, None, None] / (torch.sqrt(activ.var([0, 2, 3], unbiased=False)[None, :, None, None]) + 1e-5)
        # bias = np.zeros([activ.shape[1]])
        # weight= np.zeros([activ.shape[1]])
        # for i in range(activ.shape[1]):
        #     var= np.var(activ[:,i])
        #     bias[i] = - np.mean(activ[:,i])/ var
        # #     weight[i] = 1/ var
        # out = activ * weight[None, :, None, None] + bias[None, :, None, None]
        # for i in range(activ.shape[0]):
        #     out[i] = weight * activ[i] + bias
        # mean, std = norm.fit(out)
        # print(f'Mean and std normalized: {mean} , {std}')
        return gamma, beta


def do_cluster_init(weights, shape, previous, config, index, **kwargs):
    # cluster = {'mean' : mean, 'std': std, 'weight_stds' : weight_stds, 'components' : n_components[name]}
    dir = '/braintree/home/fgeiger/weight_initialization'
    name = f'cluster_{global_data.layers[index]}'
    pickle_in = open(f'{dir}/{name}.pkl', "rb")
    cluster = pickle.load(pickle_in)
    frac = shape[1] / weights.shape[1]
    means = cluster['mean']
    stds = cluster['std']
    weight_stds = cluster['weight_stds']
    centers = cluster['centers'].squeeze()
    new_weights = np.zeros(shape)
    for i in range(shape[0]):
        types = np.zeros([len(means)])
        for j in range(len(means)):
            types[j] = np.abs(np.round(np.random.normal(means[j], stds[j]) * frac))
        kernel = np.zeros(shape[1:])
        index = 0
        for j in range(len(types)):
            amount = types[j]
            center = centers[j]
            for k in range(int(amount)):
                if index < kernel.shape[0]:
                    noise = np.random.normal(0, weight_stds[j], center.shape)
                    channel = center + noise
                    kernel[index] = channel
                    index += 1
                else:
                    break
        while index < kernel.shape[0]:
            center = centers[-1]
            noise = np.random.normal(0, weight_stds[-1], center.shape)
            kernel[index] = center + noise
            index += 1
        random_order = random_state.permutation(shape[1])
        new_weights[i] = kernel[random_order]
    return new_weights
