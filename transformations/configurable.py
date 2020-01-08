import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def apply_nullify_small(m, config):
    percent = config[0]
    weights = m.weight.data.cpu().numpy()
    ordered = np.sort(np.abs(weights), axis=None)
    fraction = int(len(ordered) * percent)
    pivot = ordered[fraction]
    counter = -0
    for cell in np.nditer(weights, op_flags=['readwrite']):
        if np.abs(cell) <= pivot:
            cell[...] = 0.0
            counter += 1
    logger.info(f'Values set to zero: {counter} from total weights {ordered.size}, configuration:{percent}')
    m.weight.data = torch.Tensor(weights)


def apply_nullify_high(m, config):
    percent = config[0]
    weights = m.weight.data.cpu().numpy()
    ordered = np.sort(np.abs(weights), axis=None)
    fraction = int(len(ordered) * (1 - percent))
    pivot = ordered[fraction]
    counter = 0
    for cell in np.nditer(weights, op_flags=['readwrite']):
        if np.abs(cell) >= pivot:
            cell[...] = 0.0
            counter += 1
    logger.info(f'Values set to zero: {counter} from total weights {ordered.size}, configuration:{percent}')
    m.weight.data = torch.Tensor(weights)


def calculate_variances(weights, percent):
    stds = []
    for kernel_no in range(weights.shape[0]):
        stds.append(np.std(weights[kernel_no]))
    sorted_stds = np.sort(stds)
    fraction = int(len(sorted_stds) * percent)
    value = sorted_stds[fraction]
    return value, stds


def apply_high_variance_cut(m, config):
    weights = m.weight.data.cpu().numpy()
    pivot, stds = calculate_variances(weights, 1 - config[0])
    counter = 0
    for i in range(weights.shape[0]):
        if stds[i] > pivot:
            weights[i] = np.zeros(weights[i].shape)
            counter += 1
    # print(weights)
    logger.info(
        f'High variance values set to zero: {counter} from total number of kernels {weights.shape[0]}, '
        f'configuration:{config[0]}')
    m.weight.data = torch.Tensor(weights)


def apply_low_variance_cut(m, config):
    weights = m.weight.data.cpu().numpy()
    pivot, stds = calculate_variances(weights, config[0])
    counter = 0
    for i in range(weights.shape[0]):
        if stds[i] < pivot:
            weights[i] = np.zeros(weights[i].shape)
            # print(weights[i])
            counter += 1
    # print(weights)
    logger.info(
        f'Low variance values set to zero: {counter} from total number of kernels {weights.shape[0]}, configuration:'
        f'{config[0]}')
    m.weight.data = torch.Tensor(weights)
