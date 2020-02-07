import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from skimage.filters import gabor_kernel
from skimage.transform import resize
from torch.nn import init

from transformations.layer_based import random_state
from utils.correlation import auto_correlation, generate_correlation_map


def apply_to_one_layer(net, config):
    apply_to_one_layer.layer = config[0]
    apply_to_one_layer.counter = 0

    def init_weights(m):
        if type(m) == nn.Conv2d or ((type(m) == nn.Linear or type(m) == nn.BatchNorm2d) and False):
            apply_to_one_layer.counter += 1
            if apply_to_one_layer.counter is apply_to_one_layer.layer:
                function(m)

    net.apply(init_weights)
    return net


def apply_kamin(ms):
    for m in ms.modules():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data = init.kaiming_normal(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            m.weight.data = init.kaiming_normal(m.weight.data)


def apply_incremental_init(ms, config):
    kernel_values = {}
    layers = []
    for name, m in ms.named_modules():
        if type(m) == nn.Conv2d:

            if len(kernel_values) != 0:
                # m.weight.data = nn.init.xavier_normal(m.weight.data)
                if name is 'V4.conv_input':
                    means = np.add(kernel_values['V2.conv3']['means'], kernel_values['V2.skip']['means'])
                    stds = np.add(kernel_values['V2.conv3']['stds'], kernel_values['V2.skip']['stds'])
                    initialize_from_previous(m, means, stds)
                elif name is 'IT.conv_input':
                    means = np.add(kernel_values['V4.conv3']['means'], kernel_values['V2.skip']['means'])
                    stds = np.add(kernel_values['V4.conv3']['stds'], kernel_values['V2.skip']['stds'])
                    initialize_from_previous(m, means, stds)
                else:
                    means = kernel_values[layers[len(layers) - 1]]['means']
                    stds = kernel_values[layers[len(layers) - 1]]['stds']
                    initialize_from_previous(m, means, stds)
            layers.append(name)
            means = []
            stds = []
            weights = m.weight.data.cpu().numpy()
            for i in range(weights.shape[0]):
                means.append(np.mean(weights[i].flatten()))
                stds.append(np.mean(weights[i].flatten()))
            kernel_values[name] = {'means': means, 'stds': stds}
    return ms


def initialize_from_previous(m, means, stds):
    weights = m.weight.data.cpu().numpy()
    for k in range(weights.shape[0]):
        for i in range(weights.shape[1]):
            weights[k, i] = np.random.normal(np.abs(means[i]), np.abs(stds[i]), weights[k, i].shape)
    m.weight.data = torch.Tensor(weights)


def apply_fit_std_function(model, function, config):
    stds = []
    layers = []
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            layers.append(name)
            name.split('.')
            weights = m.weight.data.cpu().numpy()
            flat = weights.flatten()
            mu, std = norm.fit(flat)
            stds.append(std)
    z = np.polyfit(range(1, len(layers) + 1), stds, config[1])
    p30 = np.poly1d(np.polyfit(range(1, len(layers) + 1), stds, 1))
    p = np.poly1d(z)
    xp = np.linspace(0, 18, 100)
    # _ = plt.plot(range(1, len(layers) + 1), stds, '.', xp, p(xp), '-', xp, p30(xp), '--')
    # plt.ylim(-2, 2)
    # plt.show()
    counter = 1
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            weights = np.random.normal(0, p(counter), weights.shape)
            m.weight.data = torch.Tensor(weights)
    return model


def apply_second_layer(model, configuration):
    idx = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            if idx == 0:
                previous_weights = do_gabors(weights, configuration)
                m.weight.data = torch.Tensor(previous_weights)
            if idx == 1:
                previous_weights = do_correlation_init(weights, previous_weights)
                m.weight.data = torch.Tensor(previous_weights)
            idx += 1
    return model


def apply_gabors(model, configuration):
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            m.weight.data = torch.Tensor(do_gabors(weights, configuration))
            return model


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
    # show_kernels(weights)
    return weights


def show_kernels(weights):
    f_min, f_max = weights.min(), weights.max()
    weights = (weights - f_min) / (f_max - f_min)
    number = math.ceil(math.sqrt(weights.shape[0]))
    img = np.transpose(weights, (0, 2, 3, 1))
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


def reshape_with_project(kernel):
    omin = np.min(kernel)
    omax = np.max(kernel)
    ceiled = (kernel - omin) / (omax - omin)
    kernel = -0.25 + (ceiled * 0.5)
    return kernel
