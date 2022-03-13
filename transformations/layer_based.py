import numpy as np
import torch
import torch.nn as nn
from numpy.random.mtrand import RandomState
from scipy.stats import norm

random_state = RandomState(0)

batchnorm_shuffle = False


def apply_to_net(net, config):
    def init_weights(m):
        if type(m) == nn.Conv2d or (
                (type(m) == nn.Linear or type(m) == nn.BatchNorm2d) and batchnorm_shuffle):
            config['layer_func'](m, config)

    net.apply(init_weights)
    return net


def apply_kaiming(m, configs):
    nn.init.kaiming_normal_(m, mode='fan_out', nonlinearity='relu')


def apply_norm_dist(m, configs=None):
    weights = m.weight.data.cpu().numpy()
    flat = weights.flatten()
    mu, std = norm.fit(flat)
    torch.nn.init.normal(m.weight, mean=mu, std=std)


def apply_norm_dist_kernel(m, configs=None):
    weights = m.weight.data.cpu().numpy()
    if len(weights.shape) > 2:
        for k in range(weights.shape[0]):
            kernel_weights = weights[k].flatten()
            mu, std = norm.fit(kernel_weights)
            weights[k] = np.random.normal(mu, std, weights[k].shape)
        m.weight.data = torch.Tensor(weights)
    else:
        apply_norm_dist(m)


def apply_all_jumbler(m, configs=None):
    weights = m.weight.data.cpu().numpy()
    if len(weights.shape) > 2:
        random_order_1 = random_state.permutation(weights.shape[0])
        random_order_2 = random_state.permutation(weights.shape[1])
        random_order_3 = random_state.permutation(weights.shape[2])
        random_order_4 = random_state.permutation(weights.shape[3])
        weights = weights[random_order_1]
        weights = weights[:, random_order_2]
        weights = weights[:, :, random_order_3]
        weights = weights[:, :, :, random_order_4]
        m.weight.data = torch.Tensor(weights)
    else:
        random_order_1 = random_state.permutation(weights.shape[0])
        random_order_2 = random_state.permutation(weights.shape[1])
        weights = weights[random_order_1][:, random_order_2]
        m.weight.data = torch.Tensor(weights)


def apply_fixed_value(m, configs=None):
    weights = m.weight.data.cpu().numpy()
    new = np.full(weights.shape, 1.0)
    m.weight.data = torch.Tensor(new)


def apply_fixed_value_small(m, configs=None):
    weights = m.weight.data.cpu().numpy()
    new = np.full(weights.shape, 0.1)
    m.weight.data = torch.Tensor(new)


def apply_channel_jumbler(m, configs=None):
    weights = m.weight.data.cpu().numpy()
    if len(weights.shape) > 2:
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                random_order_1 = random_state.permutation(weights.shape[2])
                random_order_2 = random_state.permutation(weights.shape[3])
                weights[i, j] = weights[i, j, random_order_1]
                weights[i, j] = weights[i, j, :, random_order_2]
                # weights=weights[random_order_1][: , random_order_2]
                m.weight.data = torch.Tensor(weights)


def apply_in_kernel_jumbler(m, configs=None):
    weights = m.weight.data.cpu().numpy()
    if len(weights.shape) > 2:
        for i in range(weights.shape[0]):
            random_order_1 = random_state.permutation(weights.shape[1])
            random_order_2 = random_state.permutation(weights.shape[2])
            random_order_3 = random_state.permutation(weights.shape[3])
            weights[i] = weights[i][random_order_1, :, :]
            weights[i] = weights[i][:, random_order_2, :]
            weights[i] = weights[i][:, :, random_order_3]
        m.weight.data = torch.Tensor(weights)


def apply_uniform_dist(m, configs=None):
    weights = m.weight.data.cpu().numpy()
    mu, std = norm.fit(weights)
    torch.nn.init.uniform(m.weight, a=-1 * std, b=std)
