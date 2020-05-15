import os
import pickle
import sys
from os import path

import cornet
import numpy as np
import scipy as st
import tensorflow as tf
import torch
from torch import nn

from base_models import get_model, run_model_training, layers, get_config, conv_to_norm, global_data, \
    apply_generic_other
from base_models.global_data import base_dir
from base_models.test_models import get_mobilenet, get_resnet50
from base_models.trainer_performance import train
from base_models.transfer_models import create_config
from benchmark.database import create_connection, store_analysis
from utils.distributions import mixture_gaussian, best_fit_distribution
from utils.models import mobilenet_mapping, mapping_1, mobilenet_mapping_5, mobilenet_mapping_6


def measure_performance(identifier, title, do_epoch=False, do_analysis=False):
    config = get_config(identifier)
    model = get_model(identifier, False, config)
    if do_epoch:
        time = run_model_training(model, identifier, config, train)
        # time = 0
        # flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=False, print_per_layer_stat=True)
        # train_flops = flops * 2 * 3 * (1.2 * 1000000) * 1
        # print(
        #     f' Flops model {identifier}: {flops/1000000:.2f}, training flops {train_flops/1000000:.2f} params {params/1000000:.2f}')
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        path = f'{dir_path}/../scores.sqlite'
        db = create_connection(path)
        # model_id, train_time, weights_fixed, weights_to_train, additional_params, flops
        store_analysis(db, (title, time['time'], 0, 0, 0, 0))
        # store_analysis(db, (identifier, time['time'], hyp_w[-1] - hyp[-1], hyp_w[-1], hyp[-1], 0))


param_list = {}


def get_params(identifier, hyperparams=True):
    if 'hmax' in identifier:
        return 0
    if identifier in param_list:
        return param_list[identifier]
    if identifier.startswith('mobilenet'):
        if identifier.startswith('mobilenet_v1_1.0'):
            params = get_mobilenet_params()
            print(f'{identifier} has {params} parameter')
            return params
        else:
            model = get_mobilenet(f'{identifier}_random')._model
            mapping = mobilenet_mapping
            if '_v5' in identifier:
                config = get_config(identifier.split('_v5_')[1])
                mapping = mobilenet_mapping_5
            if '_v6' in identifier:
                config = get_config(identifier.split('_v6_')[1])
                mapping = mobilenet_mapping_6
            if '_v1' in identifier:
                config = get_config(identifier.split('_v1_')[1])
            if '_v7' in identifier:
                config = get_config(identifier.split('_v7_')[1])
                mapping = mobilenet_mapping_6
            config = create_config(mapping, config, model)
            if 'v5' in identifier or 'v6' in identifier:
                add_layers = ['model.2.0', 'model.2.1', 'model.6.0', 'model.6.1', 'model.12.0', 'model.12.1', 'fc',
                              'decoder']
                config['layers'] = config['layers'] + add_layers
            del config['bn_init']
            del config['batchnorm']
            model = apply_generic_other(model, config)
    elif identifier.startswith('resnet'):
        model = get_resnet50(False)
        base = identifier.split('_v3_')[1] if '_v3_' in identifier else identifier.split('_v1_')[1]
        config = get_config(base)
        config = create_config(mapping_1, config, model)
        model = apply_generic_other(model, config)
    else:
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
                print(f'layer {name} is trained, size {size}')
            elif name in config and hyperparams:
                this_mod = sys.modules[__name__]
                if identifier.startswith('mobilenet') or identifier.startswith('resnet'):
                    func = getattr(this_mod, config[config[name]].__name__)
                    idx = layers.index(config[name])
                else:
                    func = getattr(this_mod, config[name].__name__)
                params = func(m.weight.data.cpu().numpy(), config=config, index=idx)
                values += params
                hyper_params += params
                print(
                    f'layer {name} saves {size} weights and replaces it with {params} so {params / size} params')
            all += size
            idx += 1
            layer.append(name)
            all_w.append(all)
            hyp.append(hyper_params)
            hyp_w.append(values)
        if type(m) == nn.BatchNorm2d and 'batchnorm' in config:
            size = 1
            # name = config[name] if identifier.startswith('mobilenet') else name
            if name in conv_to_norm:
                for dim in np.shape(m.weight.data.cpu().numpy()): size *= dim
                if any(value in conv_to_norm[name] for value in config['layers']):
                    this_mod = sys.modules[__name__]
                    values += size
                if conv_to_norm[name] in config and hyperparams:
                    this_mod = sys.modules[__name__]
                    str(config['bn_init'])
                    func = getattr(this_mod, config['bn_init'].__name__)
                    params = func(m.weight.data.cpu().numpy(), config=config, index=idx)
                    values += params
                    hyper_params += params
        # if type(m) == nn.Linear and name is not 'decoder' and name is not 'fc':
        #     size=1
        #     for dim in np.shape(m.weight.data.cpu().numpy()): size *= dim
        #     values += size
    param_list[identifier] = values
    print(f'{identifier} has {values} parameter')
    return values


def get_mobilenet_params():
    model = get_mobilenet('mobilenet_v1_1.0_224')
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


if __name__ == '__main__':
    get_params('CORnet-S_cluster2_v2_IT_trconv3_bi')


def benchmark_epoch(identifier):
    config = get_config(identifier)
    model = get_model(identifier, False, config)
    time = run_model_training(model, identifier, False, config, train)
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
        rel = config[f'file_{index}']
        file = path.join(path.dirname(__file__), f'..{rel}')
        params = np.load(file)
    else:
        rel = config[f'file']
        file = path.join(path.dirname(__file__), f'..{rel}')
        params = np.load(file)
    param, tuples = prepare_gabor_params(params)
    # np.random.seed(0)
    components = config[f'comp_{index}'] if f'comp_{index}' in config else 0
    if components != 0:
        return components + components * param.shape[1] + components * param.shape[1] * param.shape[1]
    best_gmm = mixture_gaussian(param, weights.shape[0], components, f'gabor_{index}', analyze=True)
    return (best_gmm.weights_.size + best_gmm.means_.size + best_gmm.covariances_.size)


def do_distribution_gabor_init_channel(weights, config, index, **kwargs):
    if index != 0:
        params = np.load(config[f'file_{index}'])
    else:
        params = np.load(f'{config["file"]}')
    param, tuples = prepare_gabor_params_channel(params)
    # np.random.seed(0)
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


# def do_best_dist_init_kernel(weights, config, index, **kwargs):
#     dists = {}
#     if f'best_kernel_dist_{index}' in config:
#         name = config[f'best_kernel_dist_{index}']
#         best_dist = getattr(st, name)
#         p = best_dist.fit(weights)
#         return len(p) * weights.shape[0]


def do_in_channel_jumbler(weights, **kwargs):
    return len(weights.flatten())


def apply_to_net(net, config, **kwargs):
    count = 0
    for name, m in net.named_parameters():
        if type(m) == nn.Conv2d:
            count += config['layer_func'](m, config)
    return count


def apply_kaiming(m, configs, **kwargs):
    return 0


def apply_norm_dist(m, configs=None, **kwargs):
    return 2


def apply_norm_dist_kernel(m, configs=None, **kwargs):
    weights = m.weight.data.cpu().numpy()
    return weights.shape[0] * 2


def apply_all_jumbler(m, configs=None, **kwargs):
    weights = m.weight.data.cpu().numpy()
    return len(weights.flatten())


def apply_fixed_value(m, configs=None, **kwargs):
    return 0


def apply_fixed_value_small(m, configs=None, **kwargs):
    return 0


def do_mutual_information(m, configs=None, **kwargs):
    return 0


def do_best_dist_init_kernel(m, **kwargs):
    return 1 + (3 * m.shape[0])


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


def do_cluster_init(model, config, index, **kwargs):
    # cluster = {'mean' : mean, 'std': std, 'weight_stds' : weight_stds, 'components' : n_components[name]}
    name = f'cluster_{global_data.layers[index]}'
    pickle_in = open(f'{base_dir}/{name}.pkl', "rb")
    cluster = pickle.load(pickle_in)
    means = cluster['mean']
    stds = cluster['std']
    weight_stds = cluster['weight_stds']
    centers = cluster['centers'].squeeze()
    return len(centers.flatten()) + len(weight_stds) + len(stds) + len(means)
