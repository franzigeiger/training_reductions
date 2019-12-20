import os

import matplotlib.pyplot as plt
import numpy as np

from benchmark.database import load_scores, create_connection

benchmarks = ['dicarlo.Majaj2015.V4-pls', 'dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n',
              'fei-fei.Deng2009-top1']


def get_connection():
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    path = f'{dir_path}/../scores.sqlite'
    return create_connection(path)


def get_all_perturbations():
    return ['_jumbler', '_random', '_norm_dist', '', '_kernel_jumbler', '_norm_dist_kernel', '_channel_jumbler']


def get_all_models():
    return ['CORnet-S', 'alexnet']


def get_list_all_pert(models):
    return get_model_list(models, get_all_perturbations())


def get_list_all_models(perturbations):
    return get_model_list(get_all_models(), perturbations)


def get_model_list(models, perturbations):
    all = []
    for p in perturbations:
        for m in models:
            all.append(f'{m}{p}')
    return all


def load_data(models, benchmarks):
    db = get_connection()
    return load_scores(db, models, benchmarks)


def plot_data(benchmarks, data, labels, name, scale_fix=None):
    # res, ax = plt.subplots()
    y = np.array([0, 1, 2])
    x = np.arange(len(benchmarks))
    plt.xticks(x, labels, rotation='vertical', fontsize=8)
    # plt.yticks(y, models)
    # plt.setlabel(xlabel='Models', ylabel='Benchmarks')
    print(data)
    for key, value in data.items():
        plt.plot(x, value, label=key, linestyle="", marker="o")
    plt.legend()
    plt.tight_layout()
    if scale_fix:
        plt.ylim(scale_fix[0], scale_fix[1])
    # res.save('test.png')
    plt.savefig(f'{name}.png')
    plt.show()


def plot_data_base(data, name, x_labels=None, x_name='', y_name='', scale_fix=None, rotate=False):
    # data: {data_set_1 : [values] , data_set_2: [values]}
    if x_labels is None:
        x_labels = np.arange(len(data.values[0]))

    # data['layer']
    if rotate:
        plt.xticks(rotation='vertical', fontsize=8)
    print(data)
    for key, value in data.items():
        plt.plot(x_labels, data[key], label=key, linestyle="", marker="o")
    # plt.plot(x, data['std'], label='std', linestyle="",marker="o")
    plt.title(name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    if scale_fix:
        plt.ylim(scale_fix[0], scale_fix[1])
    plt.tight_layout()
    plt.savefig(f'{name}.png')
    plt.show()


def plot_data_map(data, name, label_field='layer', x_name='', y_name='', scale_fix=None):
    x = np.arange(len(data[label_field]))
    # data['layer']
    # plt.xticks(x, x,rotation='vertical', fontsize=8)
    print(data)
    for key, value in data.items():
        if key is not label_field:
            plt.plot(x, data[key], label=key, linestyle="", marker="o")
    # plt.plot(x, data['std'], label='std', linestyle="",marker="o")
    plt.title(name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    if scale_fix:
        plt.ylim(scale_fix[0], scale_fix[1])
    plt.tight_layout()
    plt.savefig(f'{name}.png')
    plt.show()


def plot_1_dim_data(data, name, x_name='', y_name='', scale_fix=None):
    x = np.arange(len(data))
    print(data)
    plt.plot(x, data, linestyle="", marker="o")
    plt.title(name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    if scale_fix:
        plt.ylim(scale_fix[0], scale_fix[1])
    plt.tight_layout()
    plt.savefig(f'{name}.png')
    plt.show()


def plot_histogram(data, layer, model):
    assert data.ndim == 1
    weights = np.ones(len(data)) / len(data)
    plt.hist(data, alpha=0.5, bins=100, range=(-0.4, 0.4), weights=weights, fill=True)
    plt.gca().set(title=layer, xlabel='Weight distribution')
    plt.tight_layout()
    plt.savefig(f'hist_{model}_{layer}.png')
    plt.show()


