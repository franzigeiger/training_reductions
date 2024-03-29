import numpy as np
from scipy.stats import norm
from torch import nn

from base_models.test_models import cornet_s_brainmodel
from plot.plot_data import plot_1_dim_data, plot_data_base, plot_heatmap, plot_histogram, \
    scatter_plot


def get_layer_weigh_list(random=True):
    kernel_weights = {}
    layer = []
    sizes = []
    weights = []
    model = cornet_s_brainmodel('base', (not random)).activations_model._model
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weight = m.weight.data.cpu().numpy()
            for i in range(weight.shape[0]):
                kernel_weights[f'{name}_kernel{i}'] = weight[i]
            weights.append(weight)
            layer.append(name)
            sizes.append(weight.shape[0])

    return kernel_weights, layer, sizes, weights


scales_mean = {1: (0.0, 0.4),
               4: (0.0, 0.44),
               8: (0.0, 0.75)}


def connections_mean(plot=False):
    kernel_weights, layer, sizes, weights = get_layer_weigh_list()
    influences = []
    for i in range(1, len(weights)):
        previous = weights[i - 1]
        influence_overall = np.zeros(previous.shape[0])
        for j in range(sizes[i]):
            to_analyze = np.abs(kernel_weights[f'{layer[i]}_kernel{j}'])
            for k in range(to_analyze.shape[0]):
                value = np.mean(to_analyze[k])
                influence_overall[k] = influence_overall[k] + value
        for j in range(len(influence_overall)):
            influence_overall[i] = (influence_overall[i] / sizes[i])
        if plot:
            print(layer)
            if i - 1 in scales_mean:
                plot_1_dim_data(influence_overall,
                                f'Cummulated influence per kernel from previous layer L{layer[i - 1]} to L{layer[i]}',
                                scale_fix=scales_mean[i - 1])
            else:
                plot_1_dim_data(influence_overall,
                                f'Cummulated influence per kernel from previous layer L{layer[i - 1]} to L{layer[i]}')
        influences.append(influence_overall)
    return influences, layer


scales_sum = {1: (1.5, 2.7),
              3: (2.4, 5.5),
              4: (0.6, 1.5)}


def connections(random, plot=False, normalize=False):
    kernel_weights, layer, sizes, weights = get_layer_weigh_list(random)
    influences = []
    for i in range(1, len(weights)):
        previous = weights[i - 1]
        influence_overall = np.zeros(previous.shape[0])
        for j in range(sizes[i]):
            to_analyze = np.abs(kernel_weights[f'{layer[i]}_kernel{j}'])
            sum = np.sum(to_analyze.flatten())
            dim_weights = []
            for k in range(to_analyze.shape[0]):
                value = np.sum(to_analyze[k])
                dim_weights.append(value / sum)

            for s in range(len(dim_weights)):
                influence_overall[s] = influence_overall[s] + dim_weights[s]
        if normalize:
            for s in range(len(influence_overall)):
                influence_overall[s] = influence_overall[s] / sizes[i]
        if plot:
            print(layer)
            if i - 1 in scales_sum:
                plot_1_dim_data(influence_overall,
                                f'Cummulated influence per kernel from previous layer L{layer[i - 1]} to L{layer[i]}',
                                scale_fix=scales_sum[i - 1])
            else:
                plot_1_dim_data(influence_overall,
                                f'Cummulated influence per kernel from previous layer L{layer[i - 1]} to L{layer[i]}')
        influences.append(influence_overall)
    return influences, layer


def impact_histogram(type='Sum impact', func=connections, random=True, upper_bound=0.3):
    influences, layer = func(False, False, normalize=True)
    influences_rand, layer = func(True, False, normalize=True)
    for i in range(len(influences)):
        all = influences[i]
        all_rand = influences_rand[i]
        con = np.stack((all, all_rand), axis=0)
        plot_histogram(con.T, f'{type} distribution trained and random layer {layer[i]}', bins=7,
                       labels=['Trained', 'Random'], x_axis=type)


def impact_mean_std(type='Sum impact', func=connections, random=True, upper_bound=0.3):
    influences, layer = func(random, False)
    means = []
    stds = []
    relative = []
    for i in range(len(influences)):
        inf = influences[i]
        means.append(np.mean(inf))
        stds.append(np.std(inf))
        relative.append(stds[i] / means[i])
    plot_data_base({'std': stds}, f'{type} ' + ('untrained' if random else 'trained'),
                   layer[0:(len(layer) - 1)],
                   rotate=True, scale_fix=(0.0, upper_bound))
    print(relative)


def kernel_weight_size(random=True):
    kernel_weights, layer, sizes, weights = get_layer_weigh_list(random)
    _, _, _, weights_trained = get_layer_weigh_list(False)
    layer_sum = []
    layer_sum_2 = []
    kernel_sum = []
    label = []
    for i in range(1, 17):
        layer_sum.append(np.sum(np.abs(weights[i - 1])))
        layer_sum_2.append(np.sum(np.abs(weights_trained[i - 1])))
        for j in range(sizes[i]):
            kernel_sum.append(np.sum(np.abs(kernel_weights[f'{layer[i]}_kernel{j}'])))
            label.append(f'{layer[i]}_kernel{j}')
    plot_data_base({'trained': layer_sum, 'untrained': layer_sum_2}, f'Sum of weights per layer',
                   layer[0:(len(layer) - 1)], rotate=True, scale_fix=(-100, 82))
    plot_data_base({'trained': layer_sum, 'untrained': layer_sum_2},
                   f'Absolute sum of weights per layer',
                   layer[0:(len(layer) - 1)], rotate=True, scale_fix=(0, 2000))


def impact_mean(random=True, print_plot=True, normalize=False):
    kernel_weights, layer, sizes, weights = get_layer_weigh_list(random)
    influences = []
    for i in range(1, 17):
        previous = weights[i - 1]
        previous_kernel = []
        for n in range(previous.shape[0]):
            previous_kernel.append(np.mean(np.abs(previous[n])))
        impact = []
        for j in range(sizes[i]):
            to_analyze = np.abs(kernel_weights[f'{layer[i]}_kernel{j}'])
            kernel_impact = 0.0
            for k in range(to_analyze.shape[0]):
                previous_mean = previous_kernel[k]
                value = np.mean(to_analyze[k])
                kernel_impact = kernel_impact + (previous_mean * value)
            impact.append(kernel_impact)
        name = f'Mean impact L{i - 1} to L{i}, ' + ('untrained' if random else 'trained')
        if print_plot:
            if i - 1 in scales_mean:
                plot_1_dim_data(impact, name=name,
                                x_name=f'Kernel number', y_name='Impact',
                                scale_fix=scales_mean[i - 1])
            else:
                plot_1_dim_data(impact, name=name,
                                x_name=f'Kernel number', y_name='Impact')
        influences.append(impact)
    return influences, layer


def impact_heatmap():
    kernel_weights, layer, sizes, weights = get_layer_weigh_list()
    top_k_map = {}
    limit = 20
    for i in range(1, 4):
        previous = weights[i - 1]
        previous_kernel = []
        y_axis = []
        for n in range(previous.shape[0]):
            y_axis.append(f'L{i - 1}, K{n}')
            previous_kernel.append(np.mean(np.abs(previous[n])))
        impact = []
        data = np.zeros((sizes[i], previous.shape[0]))
        x_axis = []
        for j in range(sizes[i]):
            x_axis.append(f'L{i}, K{j}')
            to_analyze = np.abs(kernel_weights[f'{layer[i]}_kernel{j}'])
            kernel_impact = 0.0
            sum = np.sum(to_analyze.flatten())
            for k in range(to_analyze.shape[0]):
                value = np.sum(to_analyze[k])
                data[j, k] = value / sum
            impact.append(kernel_impact)
        plot_heatmap(data[0:limit, 0:limit], x_axis[0:limit], y_axis[0:limit],
                     'Heat map kernel influence form previous layer sum', vmax=0.06)
    print(top_k_map)


def mean():
    kernel_weights, layer, sizes, weights = get_layer_weigh_list()
    mean_std = {'mean': [], 'std': []}
    labels = []
    for i in range(1, 17):
        previous = weights[i - 1]
        previous_kernel = []
        for n in range(previous.shape[0]):
            previous_kernel.append(np.mean(np.abs(previous[n])))
        impact = []
        for j in range(sizes[i]):
            to_analyze = np.abs(kernel_weights[f'{layer[i]}_kernel{j}'])
            kernel_impact = 0.0
            for k in range(to_analyze.shape[0]):
                previous_mean = previous_kernel[k]
                value = np.mean(to_analyze[k])
                kernel_impact = kernel_impact + (previous_mean * value)
            impact.append(kernel_impact)
        mu, std = norm.fit(impact)
        mean_std['mean'].append(mu)
        mean_std['std'].append(std)
        labels.append(f'L{i - 1}({previous.shape[0]})_to_L{i}({sizes[i]})')
    plot_data_base(mean_std, name=f'Mean+Std_with_prev_layer', x_labels=labels,
                   x_name=f'Kernel impact mean + std',
                   y_name='', rotate=True)


def conv2_conv3():
    kernel_weights, layer, sizes, weights = get_layer_weigh_list(False)
    values = {}
    values['sum_prev'] = []
    values['mean_prev'] = []
    values['std_prev'] = []
    for j in range(512):
        i = weights[5][j]
        values['mean_prev'].append(np.mean(i))
        values['std_prev'].append(np.std(i))
        values['sum_prev'].append(np.sum(i))
    values['sum'] = []
    values['mean'] = []
    values['std'] = []
    layer = weights[6].squeeze().T
    for j in range(512):
        i = layer[j]
        values['mean'].append(np.mean(i))
        values['std'].append(np.std(i))
        values['sum'].append(np.sum(i))
    scatter_plot(values['std_prev'], values['std'], x_label='Std layer 6', y_label="Std layer 7",
                 scale_fix=[0, 0.07])
    z = np.polyfit(values['std_prev'], values['std'], 1)
    p = np.poly1d(z)
    res = p(values['std_prev'])
    scatter_plot(values['std_prev'], res, x_label='Std layer 6', y_label="Std layer 7",
                 scale_fix=[0, 0.07])
    print(z)


def conv2_conv3_2():
    kernel_weights, layer, sizes, weights = get_layer_weigh_list(False)
    values = {}
    values['sum_prev'] = []
    values['mean_prev'] = []
    values['std_prev'] = []
    for j in range(512):
        i = weights[5][j]
        values['mean_prev'].append(np.mean(i))
        values['std_prev'].append(np.std(i))
        values['sum_prev'].append(np.sum(i))
    l = weights[6].squeeze()
    std_corr = []
    sum_corr = []
    for j in range(len(l)):
        i = l[j]
        std_corr.append(scatter_plot(i, values['std_prev'], x_label=f'Kernel {j} layer 7',
                                     y_label="Std layer 6"))
        sum_corr.append(scatter_plot(i, values['sum_prev'], x_label=f'Kernel {j} layer 7',
                                     y_label="Sum layer 6"))
    plot_data_base({'Sum/Avg': sum_corr, 'Std': std_corr}, 'Correlations')


def conv2():
    kernel_weights, layer, sizes, weights = get_layer_weigh_list(False)
    values = {}
    values['sum_prev'] = []
    for j in range(1):
        i = weights[5][j]
        for ch in i:
            values['sum_prev'].append(np.sum(ch))
    plot_data_base(values, 'Properties')


if __name__ == '__main__':
    conv2_conv3()
