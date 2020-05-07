import math
from heapq import nlargest

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from matplotlib.lines import Line2D
from model_tools.brain_transformation import ModelCommitment
from numpy.random.mtrand import RandomState
from scipy.stats import norm, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split
from torch import nn

from nets.pool import brain_translated_pool, base_model_pool
from nets.test_models import alexnet, cornet_s_brainmodel, get_model
from plot.plot_data import plot_data_map, plot_data_base, plot_two_scales, plot_pie, my_palette_light, plot_heatmap
from utils.cluster import cluster_data

weights = [9408, 36864, 8192, 16384, 65536, 2359296, 32768, 65536, 262144, 9437184, 262144, 131072, 262144, 1048576,
           37748736, 1048576, 512000]
sum_conv3 = 53372096 - 32768 - 131072 - 1048576 - 512000


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


def weight_init_percent(model_name, index, random=True):
    model = load_model(model_name, random)
    sum = 0
    frac = 0
    sizes = []
    labels = []
    i = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            size = 1
            for dim in np.shape(m.weight.data.cpu().numpy()): size *= dim
            sizes.append(size)
            labels.append(name)
            if i < index:
                frac += size
            sum += size
            i += 1
    # plot_pie([(frac/sum), (1- (frac/sum))], ['Fixed', 'Train'])
    print(sizes)
    print(labels)
    plot_pie(sizes, labels)


def weight_mean_std(model_name, random=False):
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


def weight_std_factor(model_name, random=False):
    model = load_model(model_name, True)
    norm_dists = {}
    # norm_dists['layer'] = []
    # norm_dists['mean'] = []
    norm_dists['value'] = []
    # pytorch model
    layers = []
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            norm_dists['value'] = []
            layers.append(name)
            name.split('.')
            weights = m.weight.data.cpu().numpy()
            n_l = weights.shape[1] * weights.shape[2] * weights.shape[2]
            for i in range(weights.shape[0]):
                flat = weights[i].flatten()
                mu, std = norm.fit(flat)
                # norm_dists['layer'].append(name)
                norm_dists['value'].append(std * 1 / 2 * n_l)
                print(f'Std: {std}')
            # plot_histogram(flat, name, model_name)
            plot_data_base(norm_dists, name, x_name='layers', y_name='Mean')


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
    model = load_model(model_name, False)
    counter = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d and counter >= 1:
            weights = m.weight.data
            # filter_weights = np.zeros((weights.shape[0], weights.shape[2], weights.shape[3]))
            weights = weights.reshape(weights.shape[0], weights.shape[1] * weights.shape[2], weights.shape[3])
            weights = weights.reshape(weights.shape[0] * weights.shape[2], weights.shape[1])
            # for i in range(weights.shape[0]):
            #     # do for all kernels
            #     filter_weights[i] = weights[i][49]
            f_min, f_max = weights.min(), weights.max()
            weights = (weights - f_min) / (f_max - f_min)
            number = math.ceil(math.sqrt(weights.shape[0]))

            # img = np.transpose(weights, (0, 1, 2))
            idx = 0
            # fig, axes = pyplot.subplots(ncols=weights.shape[0], figsize=(20, 4))
            # for j in range(8):  # in zip(axes, range(weights.shape[0])):
            #     for i in range(8):
            # if idx < img.shape[0]:
            # ax = pyplot.subplot(8, 8, idx + 1)
            # ax.set_xticks([])
            # ax.set_yticks([])
            # imgs = img[range(j*8, (j*8)+number)]
            pyplot.imshow(weights, cmap='gray')
            pyplot.title(name)
            idx += 1
            pyplot.savefig(f'{name}.png')
            pyplot.show(figsize=(20, 20))
            # return
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
                f'Filter range average {sum(ranges) / len(ranges)}, filter min average: {sum(mines) / len(maxes)}, max avg: {sum(maxes) / len(maxes)}, avg diff: {sum(all_differences) / len(all_differences)}')
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


def cluster_weights():
    import torch.nn as nn
    model = get_model('CORnet-S_base', True)
    index = 0
    prev = None
    skip = None
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.numpy()
            if weights.shape[-1] != 1:
                weights.squeeze()
                new = weights.reshape(weights.shape[0], weights.shape[1], -1)
                new = weights.reshape(new.shape[0] * new.shape[1], -1)
                kmeans = cluster_data(new, name=name)
                centers = kmeans.cluster_centers_
                centers = centers.reshape(centers.shape[0], weights.shape[2], weights.shape[2])
                centers = centers.reshape(1, centers.shape[0], centers.shape[1], centers.shape[2])
                # plot_weights(centers, name)
                labels = kmeans.labels_.reshape(weights.shape[0], weights.shape[1])
                labels = one_hot_encode(labels)
                if index > 0:
                    # predict_next(prev, labels, name)
                    predict_next_kernel_2(prev, labels, name)
                    # predict_next_full(prev, labels, name)
                prev = labels
            elif 'skip' in name:
                skip = weights.squeeze()
                predict_next(prev, weights.squeeze(), name)
            elif 'input' in name and skip is not None:
                work_prev = np.concatenate([prev, skip], axis=1)
                np.random.shuffle(work_prev)
                # predict_next(work_prev, weights.squeeze(), name)
                predict_next_kernel_2(work_prev, weights.squeeze(), name)
                # predict_next_full(work_prev, weights.squeeze(), name)
                prev = weights.squeeze()
            else:
                # predict_next(prev, weights.squeeze(), name)
                predict_next_kernel_2(prev, weights.squeeze(), name)
                # predict_next_full(prev, weights.squeeze(), name)
                prev = weights.squeeze()
            # plot_type_count(labels, name)
            #     matrix = cosine_similarity(labels)
            # plot_heatmap(matrix, 'Kernel', 'Kernel', f'{name}_similarity', vmin=0, vmax=1)
            # print(np.mean(matrix), np.std(matrix))
            index += 1


def one_hot_encode(weights):
    comp = np.max(weights) + 1
    new_weights = np.zeros([*weights.shape, comp])
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            new_weights[i, j, weights[i, j]] = 1
    return new_weights


def cluster_kernel_weights():
    import torch.nn as nn
    model = get_model('CORnet-S_base', True)
    index = 0
    prev = None
    skip = None
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.numpy()
            # if weights.shape[-1] == 1 :
            if weights.shape[-1] != 1:
                weights.squeeze()
                new = weights.reshape(weights.shape[0], weights.shape[1], -1)
                new = weights.reshape(new.shape[0], -1)

                weights.squeeze()
                new = weights.reshape(weights.shape[0], weights.shape[1], -1)
                new = weights.reshape(new.shape[0] * new.shape[1], -1)
                kmeans_channel = cluster_data(new, name=name)
                labels_channel = kmeans_channel.labels_.reshape(weights.shape[0], weights.shape[1])
                kmeans = cluster_data(labels_channel, name=name)
                centers = kmeans.cluster_centers_
                centers = centers.reshape(centers.shape[0], weights.shape[1], -1)
                centers = centers.reshape(centers.shape[0], weights.shape[1], weights.shape[2], weights.shape[2])
                labels = kmeans.labels_
                # plot_histogram(labels.flatten(), name, bins=kmeans.n_clusters)
                # mean, std =  calc_mean_std(labels)
                # weight_stds = calc_weight_std(new, kmeans.cluster_centers_, kmeans.labels_)
                # out = {'mean' : mean, 'std': std, 'weight_stds' : weight_stds, 'components' : n_components[name], 'centers' : centers}
                # pickle_out = open(f'cluster_{name}.pkl', "wb")
                # pickle.dump(out, pickle_out)
                if index > 0:
                    predict_next_kernel_typed(prev, prev_kernel, labels_channel, labels, name)
                prev = labels_channel
                prev_kernel = labels
            elif 'skip' in name:
                # skip = weights.squeeze()
                kmeans = cluster_data(weights.squeeze(), name=name)
                labels = kmeans.labels_
                predict_next_kernel_typed(prev, prev_kernel, weights.squeeze(), labels, name)
                skip_kernel = labels
                skip = weights.squeeze()
            elif 'input' in name and skip is not None:
                work_prev = np.concatenate([prev, skip], axis=1)
                work_prev_kernel = np.concatenate([prev_kernel, skip_kernel], axis=0)
                # np.random.shuffle(work_prev)
                kmeans = cluster_data(weights.squeeze(), name=name)
                labels = kmeans.labels_
                # predict_next_kernel_typed(work_prev, work_prev_kernel, weights.squeeze(),labels, name )
                prev = weights.squeeze()
                prev_kernel = labels
            else:
                kmeans = cluster_data(weights.squeeze(), name=name)
                labels = kmeans.labels_
                predict_next_kernel_typed(prev, prev_kernel, weights.squeeze(), labels, name)
                prev = weights.squeeze()
                prev_kernel = labels
            index += 1


def calc_mean_std(labels):
    bins = {}
    comp = np.max(labels) + 1
    for i in range(comp):
        bins[i] = []
    for i in labels:
        counts = np.bincount(i, minlength=comp)
        for j in range(len(counts)):
            bins[j].append(counts[j])
    means = []
    stds = []
    for key, value in bins.items():
        means.append(np.mean(value))
        stds.append(np.std(value))
    return means, stds


def calc_weight_std(weights, means, labels):
    stds = {}
    for i in range(np.max(labels) + 1):
        stds[i] = []

    for i in range(weights.shape[0]):
        true = weights[i]
        mean = means[labels[i]]
        diff = mean - true
        stds[labels[i]].append(np.std(diff))
    res_stds = []
    for k, v in stds.items():
        res_stds.append(np.mean(v))
    return res_stds


def plot_over_layer(prev, this, name):
    type = []
    for i in prev:
        counts = np.bincount(i)
        type.append(np.argmax(counts))
    index = 0
    new_this = np.zeros(this.shape)
    for i in range(np.max(prev) + 1):
        for j in range(this.shape[1]):
            if i == type[j]:
                new_this[:, index] = this[:, j]
                index += 1
    plot_heatmap(new_this, 'Channel', 'Kernel', f'{name}_sorted')


def plot_type_count(this, name):
    counts = np.zeros([this.shape[0], np.max(this) + 1])
    for i in range(this.shape[0]):
        counts[i] = np.bincount(this[i], minlength=np.max(this) + 1)
    plot_heatmap(counts, 'kernel', 'Typr', f'{name}_type_count')


def plot_correlation(prev, this, name):
    corr = []
    corr_matrix = np.zeros([this.shape[0], prev.shape[1]])
    for i in range(this.shape[0]):
        for j in range(prev.shape[1]):
            value = pearsonr(this[i], prev[:, j])[0]
            corr_matrix[i, j] = value
            # plt.plot(corr)
    # plt.title(f'{name}_correlation_channel')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.show()
    # for i in range(this.shape[0]):
    #     for j in range(prev.shape[0]):
    #         corr_matrix[i,i] = np.corrcoef(this[i], prev[j])
    plot_heatmap(corr_matrix, 'Kernel next', 'Kernel prev', f'{name}_kernel_corr')


def plot_correlation_conv2s(prev, this, name):
    corr = []
    corr_matrix = np.zeros([this.shape[0], prev.shape[0]])
    length = min(this.shape[1], prev.shape[1])
    for i in range(this.shape[0]):
        for j in range(prev.shape[0]):
            value = pearsonr(this[i][:length], prev[j][:length])[0]
            corr_matrix[i, j] = value
            # plt.plot(corr)
    # plt.title(f'{name}_correlation_channel')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.show()
    # for i in range(this.shape[0]):
    #     for j in range(prev.shape[0]):
    #         corr_matrix[i,i] = np.corrcoef(this[i], prev[j])
    plot_heatmap(corr_matrix, 'Kernel next', 'Kernel prev', f'{name}_kernel_corrconv2')


def predict_next(prev, next, name):
    reg = LinearRegression()
    if len(next.shape) > 2:
        next = np.transpose(next, [1, 0, 2]).reshape(next.shape[0], -1)
    else:
        next = next.T
    if len(prev.shape) > 2:
        prev = prev.reshape(prev.shape[0], -1)
    train, test, y_train, y_test = train_test_split(prev, next, test_size=0.1, random_state=42)
    reg.fit(train, y_train)
    y_pred = reg.predict(test)
    print(
        f'layer {name} has regression score for kernel to channel bank prediction {reg.score(train, y_train)} and explained variance {explained_variance_score(y_test, y_pred)}')


def predict_next_kernel_to_kernel(prev, next, name):
    predicatability = np.zeros([next.shape[0], prev.shape[0]])
    prev = prev.reshape(prev.shape[0], -1)
    next = next.reshape(next.shape[0], -1)
    for i in range(next.shape[0]):
        high_scores = {}
        kernel = next[i]

        # train, test, y_train, y_test = train_test_split(kernels, next, test_size=0.1, random_state=42)
        # kernels = kernels.reshape(kernels.shape[0],-1)
        next = next.reshape(next.shape[0], -1)
        best_score = -100
        best_index = 0
        min = np.min([prev.shape[0], next.shape[0]])
        kernels = np.repeat(kernel.reshape(1, -1), min, axis=0)
        for j in range(prev.shape[0]):
            reg = LinearRegression()
            reg.fit(prev[j].reshape(1, -1), kernel.reshape(1, -1))
            # score = reg.score(kernel.reshape(1, -1), next[j].reshape(1,-1))
            # print(f'Score {score}')
            score = reg.score(prev[:min], kernels)
            high_scores[j] = score
            if score > best_score:
                # print(f'Score on all: {score}')
                best_score, best_index = score, j
        best_k = nlargest(3, high_scores, key=high_scores.get)
        # if len(next.shape) >2:
        #     next = np.transpose(next, [1,0,2]).reshape(next.shape[0], -1)
        # else:
        #     next = next.T
        # if len(prev.shape) > 2:
        #     prev = prev.reshape(prev.shape[0], -1)

        # for j in range(next.shape[0]):
        # print('Best score ')
        predicatability[i, best_k] = [high_scores.get(key) for key in best_k]
        # y_pred = reg.predict(test)
        # print(f'layer {name} has regression score for kernel to channel bank prediction {reg.score(kernels, next)}' )#  and explained variance {explained_variance_score(y_test, y_pred)}
    # n_larg = nlargest(3, high_scores, key = high_scores.get)
    # reg = LinearRegression()
    # reg.fit(kernel.reshape(1, -1), next[j].reshape(1,-1))
    plot_heatmap(predicatability, 'L+1 kernel', 'L kernel', f'{name}_predictability')


def predict_next_kernel(prev, next, name):
    reg = LinearRegression()
    train, test, y_train, y_test = train_test_split(prev.T, next.T, test_size=0.5, random_state=42)
    reg.fit(train.reshape(1, -1), y_train.reshape(1, -1))
    y_pred = reg.predict(test.reshape(1, -1)).reshape(y_test.shape[0], y_test.shape[1])
    score = reg.score(train.reshape(1, -1), y_train.reshape(1, -1))
    print(
        f'layer {name} has regression score {score} and explained variance {explained_variance_score(y_test, y_pred.squeeze())}')


def predict_next_kernel_2(prev, next, name):
    reg = LinearRegression()
    min = np.min([prev.shape[0], next.shape[0]])
    score = 0
    prev = prev.reshape(prev.shape[0], -1)
    next = next.reshape(next.shape[0], -1)
    while score <= 0.5:
        rand = np.random.random_integers(0, 100)
        random_state = RandomState(rand)
        prev = prev[np.random.permutation(prev.shape[0])]
        next = next[np.random.permutation(next.shape[0])]
        train, test, y_train, y_test = train_test_split(prev[:min], next[:min], test_size=0.5, random_state=rand)
        reg.fit(train, y_train)
        y_pred = reg.predict(test)
        score1 = reg.score(train, y_train)
        score = explained_variance_score(y_test, y_pred)
        # print(f'layer {name} has regression score {score1} and explained variance {score}' )
    print(f'Found a good score for layer{name}: regression score {score1} and explained variance {score}')


def get_kernel_type(type, kernel, channel):
    return channel[np.array([i for i in range(kernel.shape[0]) if kernel.squeeze()[i] == type])]


def predict_next_kernel_typed(prev, prev_kernel, next, next_kernel, name):
    comp = np.max(prev_kernel) + 1
    comp2 = np.max(next_kernel) + 1
    if prev.dtype != np.float32:
        prev = one_hot_encode(prev).reshape(prev.shape[0], -1)
    if next.dtype != np.float32:
        next = one_hot_encode(next).reshape(next.shape[0], -1)
    for i in range(comp):
        prev_comp = get_kernel_type(i, prev_kernel, prev)
        best_type = 100
        best_score = -100
        for j in range(comp2):
            next_comp = get_kernel_type(j, next_kernel, next)
            if prev_comp.shape[0] > 1 and next_comp.shape[0] > 1:
                reg = LinearRegression()
                samples = np.min([prev_comp.shape[0], next_comp.shape[0]])
                train, test, y_train, y_test = train_test_split(prev_comp[:samples], next_comp[:samples], test_size=0.1,
                                                                random_state=42)
                reg.fit(train, y_train)
                score = reg.score(prev_comp[:samples], next_comp[:samples])
                y_pred = reg.predict(test)
                score = explained_variance_score(y_pred, y_test)
                if score > best_score:
                    best_score = score
                    best_type = j
        print(
            f'Found a good score for layer{name} of type {i}: regression score {best_score} and next layer type {best_type}')


def predict_next_full(prev, next, name):
    reg = LinearRegression()
    if len(next.shape) > 2:
        next = np.transpose(next, [1, 0, 2])
        train, test, y_train, y_test = train_test_split(prev, next, test_size=0.5, random_state=42)
    else:
        train, test, y_train, y_test = train_test_split(prev, next.T, test_size=0.5, random_state=42)
    # train, test, y_train, y_test = train, test, y_train, y_test
    reg.fit(train.reshape(1, -1), y_train.reshape(1, -1))
    y_pred = reg.predict(test.reshape(1, -1))
    y_test = y_test.reshape(1, -1)
    print(
        f'layer {name} for full kernel prediction has regression score {reg.score(train.reshape(1, -1), y_train.reshape(1, -1))} and explained variance {explained_variance_score(y_pred.squeeze(), y_test.squeeze())}')


components = [('V1.conv1', 'gabor', 4),
              ('V1.conv2', 'gabor', 1),
              ('V2.conv_input', 'kernel', 1),
              ('V2.skip', 'kernel', 1),
              ('V2.conv1', 'kernel', 1),
              ('V2.conv2', 'channel', 5),
              ('V2.conv3', 'kernel', 1),
              ('V4.conv_input', 'kernel', 1),
              ('V4.skip', 'kernel', 1),
              ('V4.conv1', 'kernel', 1),
              ('V4.conv2', 'channel', 8),
              ('V4.conv3', 'kernel', 1),
              ('IT.conv_input', 'kernel', 1),
              ('IT.skip', 'kernel', 1),
              ('IT.conv1', 'kernel', 1),
              ('IT.conv2', 'channel', 4),
              ('IT.conv3', 'kernel', 1)]


def plot_mixture_components():
    idx = 0
    colors = my_palette_light[:3]
    for tuple in components:
        if tuple[1] == 'gabor':
            plt.plot(idx, tuple[2], label=tuple[0], linestyle="", color=colors[0], marker="o")
        if tuple[1] == 'channel':
            plt.plot(idx, tuple[2], label=tuple[0], linestyle="", color=colors[1], marker="o")
        if tuple[1] == 'kernel':
            plt.plot(idx, tuple[2], label=tuple[0], linestyle="", color=colors[2], marker="o")
        idx += 1
    lines = [Line2D([0], [0], color=c, linestyle='', marker='o') for c in colors]
    labels = ['Gabor mixture', 'Channel mixture', 'Kernel mixture']
    plt.legend(lines, labels)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f'Mixture_gaussian_components.png')
    plt.show()


if __name__ == '__main__':
    # function_name = 'weight_mean_std'
    # function_name = 'kernel_weight_dist'
    # function_name = 'kernel_channel_weight_dist'
    # function_name = 'mean_compared'
    # function_name = 'visualize_kernel_sidewards'
    # function_name = 'analyze_filter_delta'
    # func = getattr(sys.modules[__name__], function_name)
    # func('CORnet-S')
    # func('alexnet')
    # func('densenet169')
    # func('resnet101')
    # visualize_second_layer('CORnet-S')
    # plot_mixture_components()
    # cluster_weights()
    cluster_kernel_weights()
    # weight_std_factor('CORnet-S')
    # weight_init_percent('CORnet-S', 5, True)
