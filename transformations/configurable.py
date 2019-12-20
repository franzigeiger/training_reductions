import numpy as np
import torch
from result_caching import store


def apply_nullify_small(m, config):
    percent = config[0]
    weights = m.weight.data.cpu().numpy()
    sorted = np.sort(np.abs(weights), axis=None)
    fraction = int(len(sorted) * percent)
    pivot = sorted[fraction]
    for cell in np.nditer(weights, op_flags=['readwrite']):
        if cell <= pivot:
            cell[...] = 0.0
    m.weight.data = torch.Tensor(weights)


def apply_nullify_high(m, config):
    percent = config[0]
    weights = m.weight.data.cpu().numpy()
    sorted = np.sort(np.abs(weights), axis=None)
    fraction = int(len(sorted) * (1 - percent))
    pivot = sorted[fraction]
    for cell in np.nditer(weights, op_flags=['readwrite']):
        if cell >= pivot:
            cell[...] = 0.0
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
    for i in range(weights.shape[0]):
        if stds[i] > pivot:
            weights[i] = np.zeros(weights[i])


def apply_low_variance_cut(m, config):
    weights = m.weight.data.cpu().numpy()
    pivot, stds = calculate_variances(weights, config[0])
    for i in range(weights.shape[0]):
        if stds[i] < pivot:
            weights[i] = np.zeros(weights[i])
