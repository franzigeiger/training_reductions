import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from model_tools.brain_transformation import ModelCommitment
from scipy.stats import norm

from model_impls.pool import brain_translated_pool, base_model_pool
from model_impls.test_models import alexnet
from plot.plot_data import plot_data_map


def load_model(model_name):
    if model_name == 'alexnet':
        return alexnet('', True)._model
    base = brain_translated_pool[model_name]
    base._ensure_loaded()
    if isinstance(base.content, ModelCommitment):
        model = base_model_pool[model_name]
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
    plot_data_map(norm_dists, model_name, scale_fix=[-0.01, 0.15])
    # model.apply(plot_distribution)


def weight_histogram(model_name):
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
    all_kernels = {}
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
                kernel_values['min'][kernel_no] = np.min(kernel_weights)
                kernel_values['max'][kernel_no] = np.max(kernel_weights)
                kernel_values['name'].append(f'{name}.kernel{kernel_no}')
                kernel_values['mean'][kernel_no] = np.mean(kernel_weights)
                kernel_values['std'][kernel_no] = np.std(kernel_weights)
            # plot_data_map(kernel_values, name, 'name')
            all_kernels['name'].append(name)
            all_kernels['max'].append(np.mean(kernel_values['min']))
            all_kernels['min'].append(np.mean(kernel_values['max']))
    plot_data_map(all_kernels, f'{model_name}.kernel.dist', 'name', 'Layer number', 'value', scale_fix=[-0.75, 1.0])


def kernel_channel_weight_dist(model_name):
    import torch.nn as nn
    model = load_model(model_name)
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
            plot_data_map(kernel_values, f'{name}_kernels', 'name', scale_fix=[-0.75, 1.0])


def visualize_first_layer(model_name):
    import torch.nn as nn
    model = load_model(model_name)
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
            for j in range(number): # in zip(axes, range(weights.shape[0])):
                for i in range(number):
                    ax = pyplot.subplot(number, number, idx+1)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    # imgs = img[range(j*8, (j*8)+number)]
                    pyplot.imshow(img[idx])
                    idx+=1
            pyplot.show()
            return


def visualize_second_layer(model_name):
    import torch.nn as nn
    model = load_model(model_name)
    counter = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d and counter >= 1:
            weights = m.weight.data.squeeze()
            filter_weights = np.zeros((weights.shape[0], weights.shape[2], weights.shape[3]))
            for i in range(weights.shape[0]):
                # do for all kernels
                filter_weights[i] = weights[i][49]
            f_min, f_max = filter_weights.min(), filter_weights.max()
            filter_weights = (filter_weights - f_min) / (f_max - f_min)
            number = math.ceil(math.sqrt(weights.shape[0]))

            img = np.transpose(filter_weights, (0, 1, 2))
            idx = 0
            # fig, axes = pyplot.subplots(ncols=weights.shape[0], figsize=(20, 4))
            for j in range(8):  # in zip(axes, range(weights.shape[0])):
                for i in range(8):
                    if idx < img.shape[0]:
                        ax = pyplot.subplot(8, 8, idx + 1)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # imgs = img[range(j*8, (j*8)+number)]
                        pyplot.imshow(img[idx], cmap='gray')
                        idx += 1
            pyplot.show(figsize=(20, 20))
            return
        elif type(m) == nn.Conv2d:
            counter += 1


if __name__ == '__main__':
    # function_name = 'weight_mean_std'
    # function_name = 'kernel_weight_dist'
    # function_name = 'kernel_channel_weight_dist'
    function_name = 'visualize_second_layer'
    func = getattr(sys.modules[__name__], function_name)
    # func('CORnet-S')
    func('alexnet')
    # func('densenet169')
    # func('resnet101')
