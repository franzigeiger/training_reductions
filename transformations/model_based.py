import cornet
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from torch.nn import init

from transformations.transformation_utils import do_fit_gabor_init, do_correlation_init, do_gabors, do_fit_gabor_dist, \
    do_correlation_init_no_reshape, do_kernel_convolution_init, do_distribution_gabor_init, do_scrumble_gabor_init, \
    layers


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
            counter += 1
            m.weight.data = torch.Tensor(weights)
    return model


def apply_second_layer_only(model, configuration):
    # Assume cornet is initialized to random weights
    trained = cornet.cornet_s(pretrained=True)
    idx = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            if idx == 0:
                mod = trained.module.V1.conv1
                m.weight.data = mod.weight.data
                previous_weights = m.weight.data.numpy()
            if idx == 1:
                previous_weights = do_correlation_init(weights, previous_weights)
                m.weight.data = torch.Tensor(previous_weights)

            idx += 1
    return model


def apply_second_layer(model, configuration):
    # second layer correlation plus first layer random gabors
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


def apply_gabor_fit_second_layer(model, configuration):
    # second layer correlation plus first layer random gabors
    idx = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            if idx == 0:
                previous_weights = do_fit_gabor_init(weights, configuration)
                m.weight.data = torch.Tensor(previous_weights)
            if idx == 1:
                previous_weights = do_correlation_init(weights, previous_weights)
                m.weight.data = torch.Tensor(previous_weights)
            idx += 1
    return model


def apply_gabors_fit(model, configuration):
    # Assume cornet is initialized to random weights
    idx = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            if idx == 0:
                m.weight.data = torch.Tensor(do_fit_gabor_init(weights, configuration))
            idx += 1
    return model


def apply_gabors(model, configuration):
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            m.weight.data = torch.Tensor(do_gabors(weights, configuration))
            return model


def apply_gabors_scrumble(model, configuration):
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            m.weight.data = torch.Tensor(do_scrumble_gabor_init(weights, configuration))
            return model


def apply_gabor_fit_second_layer_no_reshape(model, configuration):
    # second layer correlation plus first layer random gabors
    idx = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            if idx == 0:
                previous_weights = do_fit_gabor_init(weights, configuration)
                m.weight.data = torch.Tensor(previous_weights)
            if idx == 1:
                previous_weights = do_correlation_init_no_reshape(weights, previous_weights)
                m.weight.data = torch.Tensor(previous_weights)
            idx += 1
    return model


def apply_second_layer_corr_no_reshape(model, configuration):
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
                previous_weights = do_correlation_init_no_reshape(weights, previous_weights)
                m.weight.data = torch.Tensor(previous_weights)

            idx += 1
    return model


def apply_first_fit_kernel_convolution_second_layer(model, configuration):
    # second layer correlation plus first layer random gabors
    idx = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            if idx == 0:
                previous_weights = do_fit_gabor_init(weights, configuration)
                m.weight.data = torch.Tensor(previous_weights)
            if idx == 1:
                previous_weights = do_kernel_convolution_init(weights, previous_weights)
                m.weight.data = torch.Tensor(previous_weights)
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


conv_to_norm = {
    'V1.norm1': 'V1.conv1',
    'V1.norm2': 'V1.conv2',
    'V2.norm_skip': 'V2.skip',
    'V2.norm1_0': 'V2.conv1',
    'V2.norm2_0': 'V2.conv2',
    'V2.norm3_0': 'V2.conv3',
    'V2.norm1_1': 'V2.conv1',
    'V2.norm2_1': 'V2.conv2',
    'V2.norm3_1': 'V2.conv3',
    'V4.norm_skip': 'V4.skip',
    'V4.norm1_0': 'V4.conv1',
    'V4.norm2_0': 'V4.conv2',
    'V4.norm3_0': 'V4.conv3',
    'V4.norm1_1': 'V4.conv1',
    'V4.norm2_1': 'V4.conv2',
    'V4.norm3_1': 'V4.conv3',
    'V4.norm1_2': 'V4.conv1',
    'V4.norm2_2': 'V4.conv2',
    'V4.norm3_2': 'V4.conv3',
    'V4.norm1_3': 'V4.conv1',
    'V4.norm2_3': 'V4.conv2',
    'V4.norm3_3': 'V4.conv3',
    'IT.norm_skip': 'IT.skip',
    'IT.norm1_0': 'IT.conv1',
    'IT.norm2_0': 'IT.conv2',
    'IT.norm3_0': 'IT.conv3',
    'IT.norm1_1': 'IT.conv1',
    'IT.norm2_1': 'IT.conv2',
    'IT.norm3_1': 'IT.conv3',

}


def apply_generic(model, configuration):
    # second layer correlation plus first layer random gabors
    trained = cornet.cornet_s(pretrained=True, map_location=torch.device('cpu')).module
    idx = 0
    previous_weights = None
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            # weights = m.weight.data.cpu().numpy()
            if layers[idx] in configuration:
                trained_weigts = trained
                for part in name.split('.'):
                    trained_weigts = getattr(trained_weigts, part)
                trained_weigts = trained_weigts.weight.data.cpu().numpy()
                previous_weights = configuration[layers[idx]](trained_weigts, config=configuration,
                                                              previous=previous_weights,
                                                              index=idx)
                m.weight.data = torch.Tensor(previous_weights)
            idx += 1
        if type(m) == nn.BatchNorm2d and not (
                any(value in name for value in configuration['layers']) or conv_to_norm[name] in configuration[
            'layers']):
            trained_weigts = trained
            for part in name.split('.'):
                trained_weigts = getattr(trained_weigts, part)
            if 'bn_init' in configuration:
                bias = trained_weigts.bias.data.cpu().numpy()
                bn_weight = trained_weigts.weight.data.cpu().numpy()
                bn_weight = configuration['bn_init'](bn_weight, config=configuration, previous=previous_weights,
                                                     index=idx)
                bias = configuration['bn_init'](bias, config=configuration, previous=previous_weights,
                                                index=idx)
                m.weight.data = torch.Tensor(bn_weight)
                m.bias.data = torch.Tensor(bias)
            if 'batchnorm' in configuration:
                m.track_running_stats = False
                m.running_var = trained_weigts.running_var
                m.running_mean = trained_weigts.running_mean

    return model
