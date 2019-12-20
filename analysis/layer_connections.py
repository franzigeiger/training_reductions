from scipy.stats import norm
from torch import nn
import numpy as np
from model_impls.test_models import cornet_s_brainmodel

# def get_kernels():
#     if len(weights.shape) > 2:
#         for i in range(weights.shape[0]):
# random_order_1 = random_state.permutation(weights.shape[1])
# random_order_2 = random_state.permutation(weights.shape[2])
# random_order_3 = random_state.permutation(weights.shape[3])
# weights[i] = weights[i][ random_order_1, :, :]
# weights[i] = weights[i][: , random_order_2, :]
# weights[i] = weights[i][ :, :, random_order_3]
# m.weight.data = torch.Tensor(weights)
from plot.plot_data import plot_1_dim_data, plot_data_base


def get_layer_weigh_list():
    kernel_weights = {}
    layer = []
    sizes = []
    weights = []
    model = cornet_s_brainmodel('base', True).activations_model._model
    layer_number = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weight = m.weight.data.cpu().numpy()
            for i in range(weight.shape[0]):
                kernel_weights[f'{name}_kernel{i}'] = weight[i]
            weights.append(weight)
            layer.append(name)
            sizes.append(weight.shape[0])

    return kernel_weights, layer, sizes, weights


def connections():
    kernel_weights, layer, sizes, weights = get_layer_weigh_list()
    top_k_map = {}
    for i in range(1, len(weights)):
        previous = weights[i - 1]
        influencers = np.zeros(previous.shape[0])
        influence_overall = np.zeros(previous.shape[0])
        number_k = []
        for j in range(sizes[i]):
            to_analyze = np.abs(kernel_weights[f'{layer[i]}_kernel{j}'])
            sum = np.sum(to_analyze.flatten())
            dim_weights = []
            for k in range(to_analyze.shape[0]):
                value = np.sum(to_analyze[k])
                dim_weights.append(value / sum)
            # top_k = [(0,0),(0,0),(0,0),(0,0),(0,0)]
            top_k = []

            for s in range(len(dim_weights)):
                # if dim_weights[s] > top_k[0][0]:
                if dim_weights[s] > 0.02:
                    top_k.append((dim_weights[s], s))
                influence_overall[s] = influence_overall[s] + dim_weights[s]
            top_k.sort(key=lambda x: x[0])
            for tuple in top_k:
                influencers[tuple[1]] = influencers[tuple[1]] + tuple[0]
            top_k_map[f'{layer[i]}_kernel{j}'] = top_k
            number_k.append(len(top_k))

            rate = 1.0 / previous.shape[0]
        print(influencers)
        # plot_1_dim_data(number_k, 'Number of influencer per kernel, 2%')
        plot_1_dim_data(influence_overall, f'Cummulated influence per kernel from previous layer L{i-1} to L{i}')
        # plot_1_dim_data(influencers, 'Influence per kernel from previous layer in percent, 2% influence')
    print(top_k_map)


def impact():
    kernel_weights, layer, sizes, weights = get_layer_weigh_list()
    top_k_map = {}
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
        plot_1_dim_data(impact, f'In_layer_kernel_impact:_sum(k_prev.mean*k.channel.mean)L{i-1} to L{i}',
                        x_name=f'Kernel number, previous:{previous.shape[0]}', y_name='Impact', scale_fix=(0.0, 0.5))
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
        labels.append(f'L{i-1}({previous.shape[0]})_to_L{i}({sizes[i]})')
    plot_data_base(mean_std, name=f'Mean+Std_with_prev_layer', x_labels=labels, x_name=f'Kernel impact mean + std',
                   y_name='', rotate=True)


if __name__ == '__main__':
    connections()
    # mean()
