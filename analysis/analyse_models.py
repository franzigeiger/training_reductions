import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from model_tools.brain_transformation import ModelCommitment
from scipy.stats import norm

from nets.pool import brain_translated_pool, base_model_pool
from nets.test_models import alexnet, cornet_s_brainmodel
from plot.plot_data import plot_data_map, plot_data_base, plot_two_scales


def load_model(model_name, random):
    if model_name == 'alexnet':
        return alexnet('', True)._model
    base = brain_translated_pool[model_name]
    if isinstance(base.content, ModelCommitment):
        model = base_model_pool[model_name]
        model = base.layer_model.activations_model._model
    else:
        model = cornet_s_brainmodel('base', (not random)).activations_model._model
    return model


def weight_mean_std(model_name, random=False):
    import torch.nn as nn
    model = load_model(model_name, random)
    norm_dists = {}
    # norm_dists['layer'] = []
    norm_dists['mean'] = []
    norm_dists['std'] = []
    # pytorch model
    layers = []
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            layers.append(name)
            name.split('.')
            weights = m.weight.data.cpu().numpy()
            flat = weights.flatten()
            mu, std = norm.fit(flat)
            # norm_dists['layer'].append(name)
            norm_dists['mean'].append(mu)
            norm_dists['std'].append(std)
            print(f'Norm dist mean: {mu} and std: {std}')
            # plot_histogram(flat, name, model_name)
    plot_two_scales(norm_dists, model_name, x_labels=layers, x_name='layers', y_name='Mean', y_name2='Std',
                    scale_fix=[-0.01, 0.15], rotate=True)
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


def mean_var_overview(model_name, random):
    import torch.nn as nn
    model = load_model(model_name, random)
    means = []
    stds = []
    layers = []
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            layers.append(name)
            weights = m.weight.data.cpu().numpy()
            kernel_means = []
            kernel_stds = []
            for kernel_no in range(weights.shape[0]):
                kernel = weights[kernel_no]
                kernel_weights = kernel.flatten()
                kernel_means.append(np.mean(kernel_weights))
                kernel_stds.append(np.std(kernel_weights))
            means.append(np.mean(kernel_means))
            stds.append(np.mean(kernel_stds))
    plot_data_base({'means': means, 'stds': stds},
                   f'Mean and Variance of kernels ' + ('Trained' if not random else 'Untrained'), layers,
                   x_name='Layer number',
                   y_name='value', scale_fix=[-0.05, 0.2])


def mean_compared(model_name, random):
    import torch.nn as nn
    model_untrained = load_model(model_name, True)
    model_trained = load_model(model_name, False)
    means_untrained = []
    means_trained = []
    layers = []
    for name, m in model_untrained.named_modules():
        if type(m) == nn.Conv2d:
            layers.append(name)
            weights = m.weight.data.cpu().numpy()
            kernel_means = []
            for kernel_no in range(weights.shape[0]):
                kernel = weights[kernel_no]
                kernel_weights = kernel.flatten()
                kernel_means.append(np.mean(np.abs(kernel_weights)))
            means_untrained.append(np.mean(kernel_means))
    for name, m in model_trained.named_modules():
        if type(m) == nn.Conv2d:
            # layers.append(name)
            weights = m.weight.data.cpu().numpy()
            kernel_means = []
            for kernel_no in range(weights.shape[0]):
                kernel = weights[kernel_no]
                kernel_weights = kernel.flatten()
                kernel_means.append(np.mean(np.abs(kernel_weights)))
            means_trained.append(np.mean(kernel_means))
    plot_data_base({'means_untrained': means_untrained, 'means_trained': means_trained}, f'Mean trained and untrained',
                   layers, x_name='Layer number',
                   y_name='value', rotate=True)


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


def visualize_kernel_sidewards(model_name):
    import torch.nn as nn
    model = load_model(model_name, False)
    counter = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d and counter >= 1:
            weights = m.weight
            f_min, f_max = weights.min(), weights.max()
            weights = (weights - f_min) / (f_max - f_min)
            number = math.ceil(math.sqrt(weights.shape[0]))
            filter_weights = weights.data.squeeze()
            img = np.transpose(filter_weights, (0, 3, 1, 2))
            idx = 0
            for j in range(int(len(img))):
                ax = pyplot.subplot(len(img) / 4, 4, idx + 1)
                ax.set_xticks([])
                ax.set_yticks([])
                pyplot.imshow(img[idx])
                idx += 1
            pyplot.show()
            return
        elif type(m) == nn.Conv2d:
            counter += 1


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


def analyze_filter_delta(model_name):
    import torch.nn as nn
    model = load_model(model_name, False)
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.squeeze()
            ranges = []
            maxes = []
            mines = []
            all_differences = []
            for kernel in range(weights.shape[0]):
                differences = []
                deltas = [(0, 1), (0, 2), (1, 2)]
                local_rang = []
                mean_avgs = []
                for i in [0, 1, 2]:
                    filter = weights[kernel, i].numpy()
                    min = np.min(filter)
                    max = np.max(filter)
                    filter = (filter - min) / (max - min)
                    min = np.min(filter)
                    max = np.max(filter)
                    rang = (max - min)
                    mean_avgs.append((np.average(filter), np.var(filter, dtype=float), rang, min))
                    ranges.append(rang)
                    local_rang.append(rang)
                    maxes.append(max)
                    mines.append(min)
                for delta in deltas:
                    filter1 = weights[kernel, delta[0]].numpy()
                    min = np.min(filter1)
                    max = np.max(filter1)
                    filter1 = (filter1 - min) / (max - min)
                    filter2 = weights[kernel, delta[1]].numpy()
                    min = np.min(filter2)
                    max = np.max(filter2)
                    filter2 = (filter2 - min) / (max - min)
                    avg_range = (local_rang[delta[0]] + local_rang[delta[1]]) / 2
                    diff = np.average(np.true_divide(np.abs(filter1 - filter2), avg_range))
                    differences.append(diff)
                    all_differences.append(diff)
                print(f'difference in kernel {kernel}: differences: {differences}, filter means and avgs: {mean_avgs}')
            print(
                f'Filter range average {sum(ranges)/len(ranges)}, filter min average: {sum(mines)/len(maxes)}, max avg: {sum(maxes)/len(maxes)}, avg diff: {sum(all_differences)/len(all_differences)}')
            return


def analyze_signs(model_name, random):
    def weight_mean_std(model_name, random=False):
        pass

    model = load_model(model_name, random)


def analyze_normalization(model_name):
    import torch.nn as nn
    model = load_model(model_name, False)
    for name, m in model.named_modules():
        if type(m) == nn.BatchNorm2d:
            weights = m.weight.data.squeeze().numpy()
            print(f'Avg:{np.mean(weights)}, std: {np.std(weights)}')


if __name__ == '__main__':
    # function_name = 'weight_mean_std'
    # function_name = 'kernel_weight_dist'
    # function_name = 'kernel_channel_weight_dist'
    # function_name = 'mean_compared'
    # function_name = 'visualize_kernel_sidewards'
    function_name = 'analyze_filter_delta'
    func = getattr(sys.modules[__name__], function_name)
    func('CORnet-S')
    # func('alexnet')
    # func('densenet169')
    # func('resnet101')
