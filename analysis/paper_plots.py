import string

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from matplotlib import rc

from analysis.analyse_models import plot_layer_centers
from analysis.other_analysis import score_over_layers_avg as score_over_layers_resnet, models, random
from analysis.time_analysis import score_over_layers_avg, plot_models_vs, score_over_layers, \
    plot_models_benchmarks, plot_benchmarks_over_epochs, plot_first_epochs, image_scores_single, delta_heatmap
from base_models.global_data import best_special_brain_2, random_scores, no_init_conv3_train, no_gabor_conv3_train
from plot.plot_data import blue_palette
from runtime.performance import plot_num_params, plot_num_params_epochs, plot_num_params_images, \
    image_epoch_score

rc('text', usetex=True)
matplotlib.rcParams['text.latex.unicode'] = False

all_models = [no_init_conv3_train,
              best_special_brain_2]
all_names = ['TA+KN', 'TA+GC']
benchmarks = [
    'movshon.FreemanZiemba2013.V1-pls',
    'movshon.FreemanZiemba2013.V2-pls',
    'dicarlo.Majaj2015.V4-pls',
    'dicarlo.Majaj2015.IT-pls',
    'dicarlo.Rajalingham2018-i2n',
    'fei-fei.Deng2009-top1']
selection = [0, 1, 2, 3, 4]


# selection = [2, 3, 4]


def plot_figure_1_old():
    plot_benchmarks_over_epochs('CORnet-S_full',
                                (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 5, 7, 10, 15, 20),
                                benchmarks)


def plot_figure_3_old():
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    score_over_layers_avg(all_models, random_scores, all_names, imagenet=False,
                          convergence=True, ax=ax1, selection=selection)
    score_over_layers(best_special_brain_2, {}, all_names, ax=ax2, bench=benchmarks)
    score_over_layers_resnet(models, random, imagenet=False, convergence=True, ax=ax3, selection=selection)
    for n, ax in enumerate((ax1, ax2, ax3)):
        text = r'\textbf{{{letter}}}'.format(letter=string.ascii_uppercase[n])
        ax.text(-0.08, 1.04, text, transform=ax.transAxes,
                weight='semibold', size=22)
    plt.savefig(f'figure_3.svg')
    plt.show()


def plot_figure_3():
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9),
                                         gridspec_kw={'left': 0.06, 'right': 0.97, 'bottom': 0.1})
    plot_num_params(imagenet=False, entry_models=all_models, all_labels=all_names, convergence=True,
                    ax=ax1, selection=selection)
    plot_num_params(imagenet=True, entry_models=all_models, all_labels=all_names, convergence=True,
                    ax=ax2, selection=[5])
    mobilenets = {'mobilenet_v1_1.0_224': '', 'mobilenet_v7_CORnet-S_cluster2_v2_IT_trconv3_bi': '', }
    random_resnet = {'resnet_v1_CORnet-S_full': 'Standard training',
                     'resnet_v3_CORnet-S_cluster2_v2_IT_trconv3_bi': 'layer4.2.conv3_special', }
    plot_num_params(imagenet=False,
                    entry_models=[mobilenets, random_resnet, {'hmax': ''}],
                    all_labels=['MobileNet', 'Resnet50', 'Hmax'],
                    layer_random={}, convergence=True,
                    ax=ax3, selection=selection, percent=False, pal=blue_palette)
    for n, ax in enumerate((ax1, ax2, ax3)):
        text = r'\textbf{{{letter}}}'.format(letter=string.ascii_uppercase[n])
        ax.text(-0.08, 1.04, text, transform=ax.transAxes,
                weight='semibold', size=22)
    plt.tight_layout()
    plt.savefig(f'figure_3.svg')
    plt.show()


def plot_figure_5():
    fig = plt.figure(figsize=(24, 8), frameon=False)  # 'left': 0.06, 'right': 0.97, 'bottom': 0.1
    outer = gridspec.GridSpec(1, 3, left=0.06, right=0.9, bottom=0.16)
    plot_layer_centers(outer[0])
    plot_models_vs({'Cluster': {'Thin': 'CORnet-S_cluster2_v2_IT_trconv3_bi',
                                'Downstream': 'CORnet-S_cluster2_v2_IT_bi_seed42'},
                    'Mixture gaussian': {'Thin': 'CORnet-S_brain_t7_t12_wmc15_IT_bi',
                                         'Downstream': 'CORnet-S_brain_wmc15_IT_bi'},
                    'Kernel normal': {'Thin': 'CORnet-S_brain3_t7_t12_knall_IT_bi',
                                      'Downstream': 'CORnet-S_brain3_knall_IT_bi'},
                    'No gabor prior': {'Thin': 'CORnet-S_brain2_t7_t12_knall_IT_bi_v2',
                                       'Downstream': 'CORnet-S_brain2_knall_IT_bi_v2'},
                    }, 'comparison', convergence=False, gs=outer[1], selection=selection)
    models = {
        'CORnet-S_train_random': 'Kaiming Normal+Donwstream Training (KN+DT)',
        'CORnet-S_train_conv3_bi': 'Kaiming Normal+Thin Adaptation (KN+TA)',
        'CORnet-S_cluster2_v2_IT_trconv3_bi_seed42': 'Genome Compression+Thin Adaptation (GC+TA)',
        'CORnet-S_cluster2_v2_IT_bi_seed42': 'Genome Compression+Downstream Training (GC+DT)'
    }
    plot_models_benchmarks(models, 'first_generation', benchmarks, gs=outer[2])
    for n, ax in enumerate([fig.get_axes()[28], fig.get_axes()[29]], start=1):
        text = r'\textbf{{{letter}}}'.format(letter=string.ascii_uppercase[n])
        ax.text(-0.08, 1.04, text, transform=ax.transAxes,
                weight='semibold', size=22)
    ax = fig.get_axes()[0]
    ax.text(-0.5, 1.6, string.ascii_uppercase[0], transform=ax.transAxes,
            weight='semibold', size=20)
    plt.savefig(f'figure_5.svg')
    plt.show()


def plot_figure_4_2():
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 11),
                                         gridspec_kw={'left': 0.05, 'right': 0.95, 'bottom': 0.17})
    small_names = ['Genome Compression+Thin Adaptation (GC+TA)', ]
    best = {
        'CORnet-S_cluster2_v2_IT_trconv3_bi': 'IT.conv3_special',
        'CORnet-S_cluster2_v2_V4_trconv3_bi_seed42': 'V4.conv3_special',
        'CORnet-S_train_gmk1_cl2_7_7tr_bi_seed42': 'V2.conv3_special',
    }
    random = {
        'CORnet-S_full': 17,
    }
    mod = [best]
    plot_num_params_images(imagenet=False, entry_models=mod, all_labels=small_names, convergence=True,
                           images=[10000, 50000, 100000, 500000], layer_random=random,
                           selection=selection, ax=ax2)
    add_models = [no_init_conv3_train, best, no_gabor_conv3_train, {'mobilenet_v1_1.0_224': ''}, {'hmax': ''}]
    plot_first_epochs({
        'CORnet-S_full': 'Standard training',
        'CORnet-S_cluster2_v2_IT_trconv3_bi': 'AG+CT 100% Imgs',
        'CORnet-S_cluster2_v2_IT_trconv3_bi_img500000': 'AG+CT 50% Imgs',
    }
        , epochs=[0, 1, 2, 3, 6, 20], convergence=True, brain=True, ax=ax3)
    add_names = ['Kaiming Normal+Thin Adaptation (KN+CT)', 'Genome Compression+Thin Adaptation (GC+TA)',
                 'No Gabor (GC+TA no gabor)', 'Mobilenet', 'Hmax']
    small_names = ['Genome Compression+Thin Adaptation (GC+TA)', ]
    plot_num_params_epochs(imagenet=False, entry_models=mod, all_labels=small_names, convergence=True,
                           epochs=[1, 6, 20], layer_random=random,
                           selection=selection, ax=ax1)

    for n, ax in enumerate((ax1, ax2, ax3)):
        ax.text(-0.08, 1.04, string.ascii_uppercase[n], transform=ax.transAxes,
                weight='semibold', size=20)
    plt.tight_layout()
    plt.savefig(f'figure_4_2.svg')
    plt.show()


def plot_figure_4_3():
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 11),
                                         gridspec_kw={'left': 0.05, 'right': 0.95, 'bottom': 0.17})
    best = {
        'CORnet-S_cluster2_v2_IT_trconv3_bi': 'IT.conv3_special',
    }
    no_gabor_conv3_train = {
        'CORnet-S_cluster9_IT_trconv3_bi': 'IT.conv3_special',
    }
    no_init_conv3_train = {
        'CORnet-S_train_conv3_bi': 'IT.conv3_special',
    }
    best_init = {'CORnet-S_cluster2_v2_IT_bi_seed42': 'IT.conv3', }
    add_models = [no_init_conv3_train, best, no_gabor_conv3_train, best_init,
                  {'mobilenet_v1_CORnet-S_cluster2_v2_IT_trconv3_bi': ''}, {'mobilenet_v1_1.0_224': ''}, {'hmax': ''}]
    add_names = ['Kaiming Normal+Thin Adaptation (KN+TA)', 'Genome Compression+Thin Adaptation (GC+TA)',
                 'No Gabor (GC+TA no gabor)', 'Genome Compression+Downstream Training (GC+DT)', '(Mobilenet GC+TA)',
                 'Mobilenet', 'Hmax']
    layer_random = {
        'CORnet-S_full': 17,
        'CORnet-S_train_V4': 12,
        'CORnet-S_train_IT_seed_0': 6,
        'CORnet-S_train_random': 1,
    }
    plot_num_params(imagenet=False, entry_models=add_models, all_labels=add_names, convergence=True,
                    ax=ax1, selection=selection, log=True, layer_random=layer_random)
    mod = [best, no_init_conv3_train]
    small_names = ['Thin Adaptation+Genome Compression (TA+GC)', ]
    plot_num_params_images(imagenet=False, entry_models=mod, all_labels=small_names, convergence=True,
                           images=[10000, 100000, 500000],
                           selection=selection, ax=ax3, log=True, layer_random=layer_random)
    mod = [best, no_init_conv3_train]
    plot_num_params_epochs(imagenet=False, entry_models=mod, all_labels=small_names, convergence=True,
                           epochs=[1, 3, 6, 20],
                           selection=selection, ax=ax2, log=True, layer_random=layer_random)
    for n, ax in enumerate((ax1, ax2, ax3)):
        ax.text(-0.08, 1.04, string.ascii_uppercase[n], transform=ax.transAxes,
                weight='semibold', size=20)
    plt.tight_layout()
    plt.savefig(f'figure_4_3.svg')
    plt.show()


def plot_figure_3_2():
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    fig1, axes = plt.subplots(2, 2, figsize=(20, 20))
    small_names = ['Thin Adaptation+Genome Compression (TA+GC)', ]
    best = {
        'CORnet-S_cluster2_v2_IT_trconv3_bi': 'IT.conv3_special',
        'CORnet-S_cluster2_v2_V4_trconv3_bi_seed42': 'V4.conv3_special',
        'CORnet-S_train_gmk1_cl2_7_7tr_bi_seed42': 'V2.conv3_special',
    }
    random = {
        'CORnet-S_full': 17,
    }
    mod = [best]
    plot_benchmarks_over_epochs('CORnet-S_full',
                                (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 5, 7, 10, 15, 20, 43),
                                benchmarks, ax=axes[0, 0], selection=[0, 1, 2, 3, 4, 5])
    plot_num_params_epochs(imagenet=False, entry_models=mod, all_labels=small_names, convergence=True,
                           epochs=[1, 3, 6, 20],
                           selection=selection, ax=axes[0, 1], log=True, layer_random=random)
    image_scores_single('CORnet-S_full', [100, 1000, 10000, 50000, 100000, 500000], selection=[0, 1, 2, 3, 4, 5],
                        ax=axes[1, 0])
    plot_num_params_images(imagenet=False, entry_models=mod, all_labels=small_names, convergence=True,
                           images=[10000, 50000, 100000, 500000], layer_random=random,
                           selection=selection, ax=axes[1, 1])

    for n, ax in enumerate((axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])):
        ax.text(-0.08, 1.04, string.ascii_uppercase[n], transform=ax.transAxes,
                weight='semibold', size=20)
    plt.tight_layout()
    plt.savefig(f'figure_4.svg')
    plt.show()


def plot_figure_4():
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    fig1, (ax2, ax3) = plt.subplots(1, 2, figsize=(20, 10), )
    delta_heatmap('CORnet-S_cluster2_v2_IT_trconv3_bi', 'CORnet-S_full', [100, 1000, 10000, 50000, 100000, 500000],
                  [0, 1, 2, 3, 5, 6, 10, 20], selection=selection, ax=ax3)
    best = {
        'CORnet-S_full': 'Standard training',
        'CORnet-S_cluster2_v2_IT_trconv3_bi': 'TA+GC',
        'CORnet-S_train_conv3_bi': 'TA+KN',
        # 'CORnet-S_cluster2_v2_IT_bi_seed42' : 'DT+GC'
    }
    image_epoch_score(best, [100, 1000, 10000, 50000, 100000, 500000], [0, 0.2, 0.5, 0.8, 1, 3, 5, 6, 10, 20],
                      selection, ax2)
    for n, ax in enumerate((ax2, ax3)):
        text = r'\textbf{{{letter}}}'.format(letter=string.ascii_uppercase[n])
        ax.text(-0.08, 1.04, text, transform=ax.transAxes,
                weight='semibold', size=22)
    plt.tight_layout()
    plt.savefig(f'figure_4.svg')
    plt.show()


def plot_figure_1():
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    fig1, (ax1_1, ax1_2, ax2, ax3) = plt.subplots(1, 4, figsize=(24, 8),
                                                  gridspec_kw={'left': 0.06, 'right': 0.97, 'bottom': 0.1,
                                                               'width_ratios': [.1, 1, 1, 1]})
    best = {
        'CORnet-S_cluster2_v2_IT_trconv3_bi': 'IT.conv3_special',
        'CORnet-S_cluster2_v2_V4_trconv3_bi_seed42': 'V4.conv3_special',
        'CORnet-S_train_gmk1_cl2_7_7tr_bi_seed42': 'V2.conv3_special',
    }
    image_epoch_score({'CORnet-S_full': 'Standard training'}, [100, 1000, 10000, 50000, 100000, 500000],
                      [0, 0.2, 0.5, 0.8, 1, 3, 5, 6, 10, 20], selection, (ax1_1, ax1_2))
    fig1.text(0.2, 0, r'\textbf{Supervised synaptic updates} [$10^{12}$]', ha='center')  # xlabel
    plot_benchmarks_over_epochs('CORnet-S_full',
                                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 5, 7, 10, 15, 20],
                                benchmarks, ax=ax3, selection=[0, 1, 2, 3, 4, 5])
    image_scores_single('CORnet-S_full', [100, 1000, 10000, 50000, 100000, 500000], selection=[0, 1, 2, 3, 4, 5],
                        ax=ax2)
    for n, ax in enumerate((ax1_1, ax2, ax3)):
        text = r'\textbf{{{letter}}}'.format(letter=string.ascii_uppercase[n])
        ax.text(-0.08, 1.04, text, transform=ax.transAxes,
                weight='semibold', size=22)
    plt.subplots_adjust(wspace=0.04, hspace=0)
    # plt.tight_layout()
    plt.savefig(f'figure_1.svg')
    plt.show()


if __name__ == '__main__':
    # plot_figure_1()
    # plot_figure_3()
    plot_figure_4()
    plot_figure_5()
