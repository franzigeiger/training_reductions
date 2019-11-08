import sys

import matplotlib.pyplot as plt
import numpy as np
from brainscore.model_interface import BrainModel
from model_tools.brain_transformation import ModelCommitment
from scipy.stats import norm

from model_impls.pool import brain_translated_pool
from plot_data import plot_histogram, plot_data, plot_data_map


def load_model(model_name):
    base = brain_translated_pool[model_name]
    base._ensure_loaded()
    if isinstance(base.content, ModelCommitment):
        model = base.layer_model.activations_model._model
    else:
        model = base.activations_model._model
    return model


def weight_mean_std(model_name):
    import torch.nn as nn
    model = load_model(model_name)
    norm_dists = {}
    norm_dists['layer'] = []
    norm_dists['mean'] = []
    norm_dists['std'] = []
    # pytorch model
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            name.split('.')
            weights = m.weight.data.cpu().numpy()
            flat = weights.flatten()
            mu, std = norm.fit(flat)
            norm_dists['layer'].append(name)
            norm_dists['mean'].append(mu)
            norm_dists['std'].append(std)
            print(f'Norm dist mean: {mu} and std: {std}')
            # plot_histogram(flat, name, model_name)
    plot_data_map(norm_dists, model_name)
    # model.apply(plot_distribution)


def weight_histogram(model_name):
    import torch
    import torch.nn as nn
    model = load_model(model_name)
    # pytorch model
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            flat = weights.flatten()
            assert flat.ndim == 1
            weights = np.ones(len(flat)) / len(flat)
            plt.hist(flat, alpha=0.5, bins=100, range=(-0.4, 0.4), weights=weights, fill=True)
            plt.gca().set(title=name, xlabel='Weight distribution')
            plt.savefig(f'histogram{name}.png')
            plt.show()


def print_histogram(parent_name, m):
    import torch.nn as nn
    for child in m.children():
        if type(child) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            plt.hist(weights.flatten())
            plt.savefig(f'histogram{parent_name}.{m.name}.png')
        if type(child) == nn.Sequential:
            return


def kernel_weight_dist(model_name):
    import torch.nn as nn
    model = load_model(model_name)
    all_kernels =  {}
    all_kernels['name'] = []
    all_kernels['min'] = []
    all_kernels['max'] = []
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            print(f'layer:{name}, shape: {weights.shape}')
            kernel_values = {}
            kernel_values['min'] = np.empty(weights.shape[0])
            kernel_values['max'] = np.empty(weights.shape[0])
            kernel_values['name'] = []
            kernel_values['mean'] = np.empty(weights.shape[0])
            kernel_values['std'] = np.empty(weights.shape[0])
            for kernel_no in range(weights.shape[0]):
                kernel = weights[kernel_no]
                kernel_weights = kernel.flatten()
                kernel_values['min'][kernel_no] =np.min(kernel_weights)
                kernel_values['max'][kernel_no] =np.max(kernel_weights)
                kernel_values['name'].append(f'{name}.kernel{kernel_no}')
                kernel_values['mean'][kernel_no] = np.mean(kernel_weights)
                kernel_values['std'][kernel_no]=np.std(kernel_weights)
            # plot_data_map(kernel_values, name, 'name')
            all_kernels['name'].append(name)
            all_kernels['max'].append(np.mean(kernel_values['min']))
            all_kernels['min'].append(np.mean(kernel_values['max']))
    plot_data_map(all_kernels, f'{model_name}.kernel.dist', 'name', 'Layer number', 'value')

def kernel_channel_weight_dist(model_name):
    import torch.nn as nn
    model = load_model(model_name
                       )
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            print(f'layer:{name}, shape: {weights.shape}')
            kernel_values = {}
            kernel_values['min'] = []
            kernel_values['max'] = []
            kernel_values['name'] = []
            kernel_values['mean'] = []
            kernel_values['std'] = []
            for kernel_no in range(weights.shape[0]):
                kernel = weights[kernel_no]
                kernel_weights = kernel.flatten()
                kernel_values['min'].append(np.min(kernel_weights))
                kernel_values['max'].append(np.max(kernel_weights))
                kernel_values['name'].append(f'{name}.kernel{kernel_no}')
                kernel_values['mean'].append(np.mean(kernel_weights))
                kernel_values['std'].append(np.std(kernel_weights))
            plot_data_map(kernel_values, name, 'name')


if __name__ == '__main__':
    function_name = 'kernel_weight_dist'
    func = getattr(sys.modules[__name__], function_name)
    func('CORnet-S')
    func('alexnet')
    func('densenet169')
    func('resnet101')
