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
    return ['', '_random', '_jumbler', '_kernel_jumbler', '_channel_jumbler', '_norm_dist', '_norm_dist_kernel']


def get_all_models():
    return ['CORnet-S', 'alexnet', 'resnet101']


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


def plot_data_base(data, name, x_labels=None, x_name='', y_name='', scale_fix=None, rotate=False, alpha=1.0,
                   base_line=0):
    if x_labels is None:
        x_labels = np.arange(len(data.values[0]))
    if rotate:
        plt.xticks(rotation='vertical', fontsize=8)
    print(data)
    # if base_line is not 0:
    #     plt.hlines(base_line, xmin=0, xmax=1, colors='b')
    for key, value in data.items():
        if key is 'base':
            plt.plot(x_labels, data[key], label=key, linestyle="solid", marker="", alpha=alpha)
        else:
            plt.plot(x_labels, data[key], label=key, linestyle="", marker="o", alpha=alpha)
    plt.title(name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.legend()
    if scale_fix:
        plt.ylim(scale_fix[0], scale_fix[1])
    plt.tight_layout()
    file_name = name.replace(' ', '_')
    plt.savefig(f'{file_name}.png')
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


def plot_1_dim_data(data, name, x_labels=None, x_name='', y_name='', scale_fix=None):
    if x_labels is None:
        x_labels = np.arange(len(data))
    print(data)
    plt.plot(x_labels, data, linestyle="", marker="o")
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


def plot_heatmap(data, col_labels, row_labels, title, **kwargs):
    # vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
    #                              "potato", "wheat", "barley"]
    # farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
    #            "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]
    #
    # harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
    #                     [2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
    #                     [1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
    #                     [0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
    #                     [0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
    #                     [1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
    #                     [0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])

    # fig, ax = plt.subplots()
    # im = ax.imshow(data)
    #
    # # We want to show all ticks...
    # ax.set_xticks(np.arange(len(x_labels)))
    # ax.set_yticks(np.arange(len(y_labels)))
    # # ... and label them with the respective list entries
    # ax.set_xticklabels(x_labels)
    # ax.set_yticklabels(y_labels)
    #
    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")
    #
    # # Loop over data dimensions and create text annotations.
    # for i in range(len(x_labels)):
    #     for j in range(len(y_labels)):
    #         text = ax.text(j, i, data[i, j],
    #                        ha="center", va="center", color="w")
    #
    # ax.set_title(title)
    # fig.tight_layout()
    # plt.show()
    fig, ax = plt.subplots()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Impact', rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False,
    #                labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
