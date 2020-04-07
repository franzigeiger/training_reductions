import os
import sys

import cornet
import numpy as np
# from ptflops import get_model_complexity_info
import scipy as st
import torch
from torch import nn

from benchmark.database import create_connection, store_analysis
from nets import get_model, run_model_training, layers, get_config
from nets.trainer_performance import train
from plot.plot_data import plot_data_base
from utils.distributions import mixture_gaussian, best_fit_distribution


def measure_performance(identifier, title, do_epoch=False):
    config = get_config(identifier)
    model = get_model(identifier, False, config)
    values = 0
    hyper_params = 0
    all = 0
    hyp = []
    hyp_w = []
    all_w = []
    layer = []
    idx = 0
    for name, m in model.named_modules():
        if type(m) == torch.nn.Conv2d:
            size = 1
            for dim in np.shape(m.weight.data.cpu().numpy()): size *= dim
            if any(value in name for value in config['layers']):
                values += size
            else:
                this_mod = sys.modules[__name__]
                str(config[name])
                func = getattr(this_mod, config[name].__name__)
                params = func(m.weight.data.cpu().numpy(), config=config, index=idx)
                values += params
                hyper_params += params
            all += size
            idx += 1
            layer.append(name)
            all_w.append(all)
            hyp.append(hyper_params)
            hyp_w.append(values)

    plot_data_base({'Total weights': all_w, 'Unfrozen values': hyp_w, 'Distribution Parameters': hyp},
                   f'Weight compression for {title}',
                   layer[0:(len(layer))], rotate=True, y_name='Num. of parameters')
    if do_epoch:
        time = run_model_training(identifier, False, config, train)
        # flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        path = f'{dir_path}/../scores.sqlite'
        db = create_connection(path)
        # model_id, train_time, weights_fixed, weights_to_train, additional_params, flops
        store_analysis(db, (identifier, time['time'], hyp_w[-1] - hyp[-1], hyp_w[-1], hyp[-1], 0))


def benchmark_epoch(identifier):
    config = get_config(identifier)
    model = get_model(identifier, False, config)
    time = run_model_training(identifier, False, config, train)
    # flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)


def do_fit_gabor_dist(weights, config, **kwargs):
    # fit independent gaussians for each gabor parameter
    params = np.load(config["file"])
    return params.shapep[2] - 1 * 2


def do_fit_gabor_init(weights, config, **kwargs):
    gabor_params = np.load(config["file"])
    return gabor_params[0] * gabor_params[1] * 7


def do_kernel_normal_distribution_init(weights, **kwargs):
    return weights.shape[0] * 2


def do_layer_normal_distribution_init(weights, **kwargs):
    return 2


def do_scrumble_gabor_init(weights, config, **kwargs):
    # 3 * gabor filter per kernel
    return 24 * weights.shape[0]


def do_gabors(weights, config, **kwargs):
    return 1


def do_correlation_init(weights, previous, **kwargs):
    return 0


def do_correlation_init_no_reshape(weights, previous, **kwargs):
    return 0


def do_kernel_convolution_init(weights, previous, **kwargs):
    return 0


def do_distribution_gabor_init(weights, config, index, **kwargs):
    if index != 0:
        params = np.load(config[f'file_{index}'])
    else:
        params = np.load(f'{config["file"]}')
    param, tuples = prepare_gabor_params(params)
    np.random.seed(0)
    components = config[f'comp_{index}'] if f'comp_{index}' in config else 0
    if components != 0:
        return components + components * param.shape[1], components * param.shape[1] * param.shape[1]
    best_gmm = mixture_gaussian(param, weights.shape[0], components, f'gabor_{index}', analyze=True)
    return len(best_gmm.weights_.flatten()) + len(best_gmm.means_.flatten()) + len(best_gmm.covariances_.flatten())


def do_distribution_gabor_init_channel(weights, config, index, **kwargs):
    if index != 0:
        params = np.load(config[f'file_{index}'])
    else:
        params = np.load(f'{config["file"]}')
    param, tuples = prepare_gabor_params_channel(params)
    np.random.seed(0)
    components = config[f'comp_{index}'] if f'comp_{index}' in config else 0
    if components != 0:
        return components + components * param.shape[1], components * param.shape[1] * param.shape[1]
    best_gmm = mixture_gaussian(param, weights.shape[0], components, f'gabor_{index}', analyze=True)
    return len(best_gmm.weights_.flatten()) + len(best_gmm.means_.flatten()) + len(best_gmm.covariances_.flatten())


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
    if components != 0:
        return components + components * params.shape[1] + components * params.shape[1] * params.shape[1]

    best_gmm = mixture_gaussian(params, params.shape[0], components, f'weight_dim{dim}_{index}', analyze=True)
    return len(best_gmm.weights_.flatten()) + len(best_gmm.means_.flatten()) + len(best_gmm.covariances_.flatten())


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


def do_best_dist_init_layer(weights, config, index, **kwargs):
    if f'best_layer_dist_{index}' in config:
        name = config[f'best_layer_dist_{index}']
        best_dist = getattr(st, name)
        p = best_dist.fit(weights)
        return len(p)
    name, params = best_fit_distribution(weights)
    print(f'Best fit distribution: {name}, params: {params}')
    return len(params)


def do_best_dist_init_kernel(weights, config, index, **kwargs):
    dists = {}
    if f'best_kernel_dist_{index}' in config:
        name = config[f'best_kernel_dist_{index}']
        best_dist = getattr(st, name)
        p = best_dist.fit(weights)
        return len(p) * weights.shape[0]


def do_in_channel_jumbler(weights, **kwargs):
    return len(weights.flatten())


def apply_to_net(net, config):
    count = 0
    for name, m in net.named_parameters():
        if type(m) == nn.Conv2d:
            count += config['layer_func'](m, config)
    return count


def apply_kaiming(m, configs):
    return 0


def apply_norm_dist(m, configs=None):
    return 2


def apply_norm_dist_kernel(m, configs=None):
    weights = m.weight.data.cpu().numpy()
    return weights.shape[0] * 2


def apply_all_jumbler(m, configs=None):
    weights = m.weight.data.cpu().numpy()
    return len(weights.flatten())


def apply_fixed_value(m, configs=None):
    return 0


def apply_fixed_value_small(m, configs=None):
    return 0


def apply_channel_jumbler(m, configs=None):
    weights = m.weight.data.cpu().numpy()
    return len(weights.flatten())


def apply_in_kernel_jumbler(m, configs=None):
    weights = m.weight.data.cpu().numpy()
    return len(weights.flatten())


def apply_uniform_dist(m, configs=None):
    return 1


def apply_to_one_layer(net, config):
    return 0


def apply_kamin(ms):
    return 0


def apply_incremental_init(ms, config):
    return 0


def apply_fit_std_function(model, function, config):
    layers = []
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            layers.append(name)
    return len(layers)


def apply_second_layer_only(model, configuration):
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            previous_weights = m.weight.data.numpy()
            return len(previous_weights.flatten())


def apply_second_layer(model, configuration):
    # second layer correlation plus first layer random gabors
    idx = 0
    count = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            if idx == 0:
                count += do_gabors(weights, configuration)
            if idx == 1:
                count += do_correlation_init(weights, None)
            count


def apply_gabor_fit_second_layer(model, configuration):
    # second layer correlation plus first layer random gabors
    idx = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            if idx == 0:
                return do_fit_gabor_init(weights, configuration)


def apply_gabors_fit(model, configuration):
    # Assume cornet is initialized to random weights
    idx = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            return do_fit_gabor_init(weights, configuration)


def apply_gabors(model, configuration):
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            return do_fit_gabor_init(do_gabors(weights, configuration), configuration)


def apply_gabors_scrumble(model, configuration):
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            return do_fit_gabor_init(do_scrumble_gabor_init(weights, configuration), configuration)


def apply_gabor_fit_second_layer_no_reshape(model, configuration):
    # second layer correlation plus first layer random gabors
    idx = 0
    counter = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            if idx == 0:
                counter += do_fit_gabor_init(weights, configuration)
            if idx == 1:
                return counter + do_correlation_init_no_reshape(weights, None)
            idx += 1


def apply_second_layer_corr_no_reshape(model, configuration):
    trained = cornet.cornet_s(pretrained=True)
    idx = 0
    counter = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            if idx == 0:
                counter += len(m.weight.data.cpu().numpy().flatten())
            if idx == 1:
                return counter + do_correlation_init_no_reshape(weights, None)
            idx += 1


def apply_first_fit_kernel_convolution_second_layer(model, configuration):
    # second layer correlation plus first layer random gabors
    idx = 0
    counter = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            if idx == 0:
                counter += do_fit_gabor_init(weights, configuration)
            if idx == 1:
                return counter + do_kernel_convolution_init(weights, None)
            idx += 1
    return model


def apply_kernel_convolution_second_layer(model, configuration):
    trained = cornet.cornet_s(pretrained=True)
    idx = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            if idx == 0:
                mod = trained.module.V1.conv1
                m.weight.data = mod.weight.data
                previous_weights = m.weight.data.cpu().numpy()
            if idx == 1:
                previous_weights = do_kernel_convolution_init(weights, previous_weights)
                m.weight.data = torch.Tensor(previous_weights)
            idx += 1
    return model


def apply_first_dist_kernel_convolution_second_layer(model, configuration):
    # second layer correlation plus first layer random gabors
    idx = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            if idx == 0:
                previous_weights = do_distribution_gabor_init(weights, configuration, idx)
                m.weight.data = torch.Tensor(previous_weights)
            if idx == 1:
                previous_weights = do_kernel_convolution_init(weights, previous_weights)
                m.weight.data = torch.Tensor(previous_weights)
            idx += 1
    return model


def apply_gabors_dist(model, configuration):
    # Assume cornet is initialized to random weights
    idx = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            if idx == 0:
                m.weight.data = torch.Tensor(do_distribution_gabor_init(weights, configuration, idx))
            idx += 1
    return model


def apply_gabor_dist_second_layer_no_reshape(model, configuration):
    # second layer correlation plus first layer random gabors
    idx = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            if idx == 0:
                previous_weights = do_distribution_gabor_init(weights, configuration, idx)
                m.weight.data = torch.Tensor(previous_weights)
            if idx == 1:
                previous_weights = do_correlation_init_no_reshape(weights, previous_weights)
                m.weight.data = torch.Tensor(previous_weights)
            idx += 1
    return model


def apply_gabors_dist_old(model, configuration):
    # Assume cornet is initialized to random weights
    idx = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            if idx == 0:
                m.weight.data = torch.Tensor(do_fit_gabor_dist(weights, configuration))
            idx += 1
    return model


def apply_generic(model, configuration):
    # second layer correlation plus first layer random gabors
    idx = 0
    previous_weights = None
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            if layers[idx] in configuration:
                previous_weights = configuration[layers[idx]](weights, config=configuration, previous=previous_weights,
                                                              index=idx)
                m.weight.data = torch.Tensor(previous_weights)
            idx += 1
    return model


if __name__ == '__main__':
    if len(sys.argv) > 1 is not None:
        measure_performance(sys.argv[1], '6 layers(V1 & V2 - V2.conv3), Imagenet focus', True)
    # Best V2 model Imagenet:
    # measure_performance('CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_wm6_full', '6 layers(V1 & V2 - V2.conv3)')
    # Best V2 model Brain benchmarks:
    # measure_performance('CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_full','6 layers(V1 & V2 - V2.conv3), Brain focus')
    # # Best V4 model Imagenet:
    measure_performance('CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12', '10 layers(V1,V2 & V4 - V2.conv3 & V4.conv3)')
    # # Best V4 model Brain benchmarks:
    # measure_performance('CORnet-S_train_kn8_kn9_kn10_wmk11_kn12', '12 layers(V1,V2 & V4 - V2.conv3 & V4.conv3), Brain focus')
