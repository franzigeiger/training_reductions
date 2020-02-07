import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

from benchmark.database import load_scores, create_connection

benchmarks = ['dicarlo.Majaj2015.V4-pls', 'dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n',
              'fei-fei.Deng2009-top1']

benchmarks_small = ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n']


def get_connection(name='scores'):
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    path = f'{dir_path}/../{name}.sqlite'
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


def load_data_openmind(models, benchmarks):
    db = get_connection('scores_openmind')
    return load_scores(db, models, benchmarks)


def plot_data(benchmarks, data, labels, name, scale_fix=None):
    sns.set()
    sns.set_context("paper")
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
                   x_ticks=None, log=False):
    sns.set()
    sns.set_context("paper")
    if x_labels is None:
        x_labels = np.arange(len(data.values[0]))
    if rotate:
        plt.xticks(rotation='vertical', fontsize=8)
    print(data)
    # if base_line is not 0:
    #     plt.hlines(base_line, xmin=0, xmax=1, colors='b')
    for key, value in data.items():
        if key in ['base', 'base_untrained', 'base_trained']:
            plt.plot(x_labels, data[key], label=key, linestyle="solid", marker="", alpha=alpha)
        else:
            plt.plot(x_labels, data[key], label=key, scalex=True, linestyle="-", marker=".", alpha=alpha)
    plt.title(name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    if log:
        plt.xscale('symlog')
        ax = plt.gca()
        ax.xaxis.set_major_formatter(ScalarFormatter())
    if x_ticks:
        plt.xticks(x_ticks)
    plt.legend()
    if scale_fix:
        plt.ylim(scale_fix[0], scale_fix[1])
    plt.tight_layout()
    file_name = name.replace(' ', '_')
    plt.savefig(f'{file_name}.png')
    plt.show()


def plot_two_scales(data, name, x_labels=None, x_name='', y_name='', y_name2='', scale_fix=None, rotate=False,
                    alpha=1.0,
                    base_line=0):
    if x_labels is None:
        x_labels = np.arange(len(data.values[0]))
    if rotate:
        plt.xticks(rotation='vertical', fontsize=8)
    print(data)
    # if base_line is not 0:
    #     plt.hlines(base_line, xmin=0, xmax=1, colors='b')
    isFirst = True
    fig, ax1 = plt.subplots()
    for key, value in data.items():
        if isFirst:
            ax1.set_ylabel(y_name, color='blue')
            ax1.plot(x_labels, data[key], linestyle="", marker="o", alpha=alpha)
            ax1.tick_params(axis='y')
            ax1.tick_params(labelrotation=45)
            isFirst = False
        else:
            ax2 = plt.twinx()
            ax2.plot(x_labels, data[key], linestyle="", marker="o", alpha=alpha, color='orange')
            ax2.set_ylabel(y_name2, color='orange')
            ax2.tick_params(axis='y', color='orange')
    plt.title(name)
    ax1.set_xlabel(x_name)
    # plt.ylabel(y_name)
    plt.legend()
    # if scale_fix:
    #     plt.ylim(scale_fix[0], scale_fix[1])
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
    file_name = name.replace(' ', '_')
    plt.savefig(f'{file_name}.png')
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


def plot_histogram(data, name, bins=100, labels=[], x_axis='Weight distribution'):
    plt.hist(data, alpha=0.5, bins=bins, fill=True, histtype='bar', density=True, label=labels)
    plt.legend(prop={'size': 10})
    plt.gca().set(title=name, xlabel=x_axis)
    plt.tight_layout()
    file_name = name.replace(' ', '_')
    plt.savefig(f'{file_name}.png')
    plt.show()


def plot_heatmap(data, col_labels, row_labels, title, **kwargs):
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
