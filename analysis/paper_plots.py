import string

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from matplotlib import gridspec
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

from analysis.analyse_models import plot_layer_centers
from analysis.time_analysis import plot_models_vs, plot_models_benchmarks, plot_benchmarks_over_epochs, \
    image_scores_single, delta_heatmap, \
    image_epoch_heatmap
from base_models.global_data import no_init_conv3_train
from plot.plot_data import blue_palette
from runtime.performance import plot_num_params, image_epoch_score

matplotlib.rcParams['text.latex.unicode'] = False
rc('font', **{'family': 'Arial', 'serif': ['Arial']})
rc('text', usetex=True)
plt.rcParams['svg.fonttype'] = 'none'

all_models = [no_init_conv3_train, ]
all_names = ['Critical Training', 'TA+WC']
benchmarks = [
    'movshon.FreemanZiemba2013.V1-pls',
    'movshon.FreemanZiemba2013.V2-pls',
    'dicarlo.Majaj2015.V4-pls',
    'dicarlo.Majaj2015.IT-pls',
    'dicarlo.Rajalingham2018-i2n',
    'fei-fei.Deng2009-top1']
selection = [0, 1, 2, 3, 4]


# selection = [2, 3, 4]


def plot_figure_1():
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8),
                                         gridspec_kw={'left': 0.06, 'right': 0.97, 'bottom': 0.1,
                                                      })
    plot_benchmarks_over_epochs('CORnet-S_full',
                                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 5, 7, 10, 15, 20],
                                benchmarks, ax=ax2, selection=[0, 1, 2, 3, 4])
    image_scores_single('CORnet-S_full', [100, 1000, 10000, 50000, 100000, 500000], selection=[0, 1, 2, 3, 4],
                        ax=ax3)
    image_epoch_score({'CORnet-S_full': 'Standard training'}, [100, 1000, 10000, 50000, 100000, 500000],
                      [0.2, 0.5, 0.8, 1, 3, 5, 6, 10, 20], selection, [ax1], with_weights=False, make_trillions=False)
    fig1.text(0.2, 0, r'\textbf{Supervised updates} (Training Epochs x Labeled Images)', ha='center', size=22)  # xlabel
    for n, ax in enumerate((ax1, ax2, ax3)):
        text = r'\textbf{{{letter}}}'.format(letter=string.ascii_uppercase[n])
        ax.text(-0.08, 1.04, text, transform=ax.transAxes,
                weight='semibold', size=24)
    plt.tight_layout()
    plt.savefig(f'figure_1.svg', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_figure_2():
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    fig = plt.figure(figsize=(12, 8), frameon=False)
    outer = gridspec.GridSpec(1, 3, left=0.06, right=0.9, bottom=0.18)
    plot_layer_centers(outer[2])
    ax1 = plt.subplot(outer[0, :2])
    ax2 = plt.subplot(outer[0, 2])
    plot_models_vs({'': {r'\textbf{Kaiming \\ Normal}': 'CORnet-S_full',
                         r'\textbf{Weight \\ Compression}': 'CORnet-S_cluster2_v2_IT_bi'}, }, '', selection=selection,
                   ax=ax2, imagenet=False, convergence=False)
    im = Image.open('/plot_output/gc.png')
    ax1.imshow(im)
    ax1.set_axis_off()
    for n, ax in enumerate((ax1, ax2)):
        text = r'\textbf{{{letter}}}'.format(letter=string.ascii_uppercase[n])
        ax.text(-0.08, 1.04, text, transform=ax.transAxes,
                weight='semibold', size=24)
    plt.tight_layout()
    plt.savefig(f'figure_2.svg', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_figure_3():
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    fig = plt.figure(figsize=(16, 16), frameon=False)
    grid = plt.GridSpec(2, 2, left=0.08, right=0.9, bottom=0.16)
    ax1 = plt.subplot(grid[0, 1])
    ax2 = plt.subplot(grid[1, 0])
    ax3 = plt.subplot(grid[1, 1])
    ax0 = plt.subplot(grid[0, 0])
    plot_num_params(imagenet=False, entry_models=all_models, all_labels=all_names, convergence=True,
                    ax=ax2, selection=selection)
    plot_num_params(imagenet=True, entry_models=all_models, all_labels=all_names, convergence=True,
                    ax=ax3, selection=[5])

    im = Image.open('/plot_output/circuit.png')
    ax1.imshow(im)
    ax1.set_axis_off()

    im = Image.open('/plot_output/train3.png')
    ax0.imshow(im)
    ax0.set_axis_off()
    for n, ax in enumerate((ax0, ax1, ax2, ax3)):
        text = r'\textbf{{{letter}}}'.format(letter=string.ascii_uppercase[n])
        ax.text(-0.08, 1.04, text, transform=ax.transAxes,
                weight='semibold', size=24)
    plt.tight_layout()
    plt.savefig(f'figure_3.svg', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_figure_4():
    sns.set_style("whitegrid", {'grid.color': '.95', })
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8),
                                    gridspec_kw={'left': 0.06, 'right': 0.97, 'bottom': 0.1,
                                                 })

    divider = make_axes_locatable(ax1)
    ax1_2 = divider.new_horizontal(size="1800%", pad=0.1)
    fig1.add_axes(ax1_2)
    delta_heatmap('CORnet-S_cluster2_v2_IT_trconv3_bi', 'CORnet-S_full', [100, 1000, 10000, 50000, 100000, 500000],
                  [0, 1, 3, 6, 10, 20], selection=selection,
                  title=r'\textbf{$\Delta$ Brain Predictivity} [percent points]', ax=ax2)
    best = {
        'CORnet-S_full': 'Fewer supervised updates',
        'CORnet-S_cluster2_v2_IT_trconv3_bi': '+ WC+CT',
        'mobilenet_v1_1.0_224': 'Mobilenet',
        'resnet_v1_CORnet-S_full': 'Resnet',
        'hmax': 'Hmax',
    }
    image_epoch_score(best, [100, 1000, 10000, 50000, 100000, 500000], [0, 0.2, 0.5, 0.8, 1, 3, 5, 6, 10, 20],
                      selection, [ax1, ax1_2], make_trillions=False, legend=True, log=True)
    ax1_2.text(0.2, 0, r'\textbf{Supervised synaptic updates}\\(training epochs x labeled images x trained synapses)',
               ha='center', size=22)
    for n, ax in enumerate((ax1, ax2)):
        text = r'\textbf{{{letter}}}'.format(letter=string.ascii_uppercase[n])
        ax.text(-0.08, 1.04, text, transform=ax.transAxes,
                weight='semibold', size=24)
    plt.savefig(f'figure_4.svg', bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_figure_5():
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    fig = plt.figure(figsize=(24, 8), frameon=False)  # 'left': 0.06, 'right': 0.97, 'bottom': 0.1
    outer = gridspec.GridSpec(1, 3, left=0.06, right=0.9, bottom=0.18)
    plot_layer_centers(outer[2])
    ax0 = plt.subplot(outer[0])
    models = {
        'CORnet-S_train_random': 'KN',
        'CORnet-S_cluster2_v2_IT_bi_seed42': 'WC',
        'CORnet-S_train_conv3_bi': 'KN+CT',
        'CORnet-S_cluster2_v2_IT_trconv3_bi_seed42': 'WC+CT',
    }
    ax1 = plt.subplot(outer[1])
    plot_models_benchmarks(models, 'first_generation', benchmarks, ax=ax1)

    mobilenets = {'mobilenet_v1_1.0_224': '', 'mobilenet_v6_CORnet-S_cluster2_v2_IT_trconv3_bi': '', }
    random_resnet = {'resnet_v1_CORnet-S_full': 'Standard training',
                     'resnet_v3_CORnet-S_cluster2_v2_IT_trconv3_bi': 'layer4.2.conv3_special', }
    plot_num_params(imagenet=False,
                    entry_models=[mobilenets, random_resnet, ],
                    all_labels=['MobileNet', 'Resnet50'],
                    layer_random={}, convergence=True, ylim=[0, 0.6],
                    ax=ax0, selection=selection, percent=False, pal=blue_palette[2:])
    # ax1 = plt.subplot(outer[0])
    ax = fig.get_axes()[0]
    ax.text(-0.5, 1.6, string.ascii_uppercase[2], transform=ax.transAxes,
            weight='semibold', size=24)
    for n, ax in enumerate([ax0, ax1], start=0):
        text = r'\textbf{{{letter}}}'.format(letter=string.ascii_uppercase[n])
        ax.text(-0.08, 1.04, text, transform=ax.transAxes,
                weight='semibold', size=24)
    fig.tight_layout()
    plt.savefig(f'figure_5.svg', bbox_inches='tight', pad_inches=0)
    plt.show()


def supp_4():
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10), )
    image_epoch_heatmap('CORnet-S_full', [100, 1000, 10000, 50000, 100000, 500000],
                        [0, 1, 2, 3, 5, 6, 10, 20], title=r'\textbf{Standard training epochs/images scores}',
                        selection=selection, ax=ax3)
    image_epoch_heatmap('CORnet-S_cluster2_v2_IT_trconv3_bi', [100, 1000, 10000, 50000, 100000, 500000],
                        [0, 1, 2, 3, 5, 6, 10, 20], title=r'\textbf{WC + CT epochs/images scores}', selection=selection,
                        ax=ax2)
    divider = make_axes_locatable(ax1)
    ax1_2 = divider.new_horizontal(size="1800%", pad=0.1)
    fig1.add_axes(ax1_2)
    best = {
        'CORnet-S_full': 'Fewer supervised updates',
        'CORnet-S_cluster2_v2_IT_trconv3_bi': '+ WC+CT',
        'mobilenet_v1_1.0_224': 'Mobilenet',
        'resnet_v1_CORnet-S_full': 'Resnet',
        'hmax': 'Hmax',
        'CORnet-S_train_conv3_bi': '+ CT',
        # 'CORnet-S_cluster2_v2_IT_bi_seed42': 'WC + DT',
    }
    image_epoch_score(best, [100, 1000, 10000, 50000, 100000, 500000], [0, 0.2, 0.5, 0.8, 1, 3, 5, 6, 10, 20],
                      selection, (ax1, ax1_2), legend=True)
    # for n, ax in enumerate((ax1)):
    text = r'\textbf{{{letter}}}'.format(letter=string.ascii_uppercase[0])
    fig1.text(0.2, 0, r'\textbf{Supervised synaptic updates}\\(training epochs x labeled images x trained synapses)',
              ha='center', size=22)  # xlabel
    for n, ax in enumerate((ax1, ax2, ax3)):
        text = r'\textbf{{{letter}}}'.format(letter=string.ascii_uppercase[n])
        ax.text(-0.08, 1.04, text, transform=ax.transAxes,
                weight='semibold', size=24)
    # plt.tight_layout()
    plt.savefig(f'supp_4.svg')
    plt.show()


def supp_5():
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    fig1, (ax3) = plt.subplots(1, 1, figsize=(10, 8), )
    plot_models_vs({'Cluster': {
        'No training': 'CORnet-S_cluster2_v2_IT_bi_seed42',
        'Critical training': 'CORnet-S_cluster2_v2_IT_trconv3_bi', },
        'Mixture gaussian': {'No training': 'CORnet-S_brain_wmc15_IT_bi',
                             'Critical training': 'CORnet-S_brain_t7_t12_wmc15_IT_bi',
                             },
        'Kernel normal': {'No training': 'CORnet-S_brain3_knall_IT_bi',
                          'Critical training': 'CORnet-S_brain3_t7_t12_knall_IT_bi',
                          },
        'No gabor prior': {'No training': 'CORnet-S_brain2_knall_IT_bi_v2',
                           'Critical training': 'CORnet-S_brain2_t7_t12_knall_IT_bi_v2',
                           },
    }, 'comparison', convergence=False, ax=ax3, epoch=6, selection=selection)
    plt.tight_layout()
    plt.savefig(f'supp_5.svg', bbox_inches='tight', pad_inches=0)
    plt.show()


def supp_6():
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), )
    plot_benchmarks_over_epochs('CORnet-S_cluster2_v2_IT_trconv3_bi',
                                [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 5, 10, 20],
                                benchmarks, selection=[0, 1, 2, 3, 4], ax=ax1)
    image_scores_single('CORnet-S_cluster2_v2_IT_trconv3_bi', [100, 1000, 10000, 50000, 100000, 500000],
                        selection=[0, 1, 2, 3, 4],
                        ax=ax2)

    for n, ax in enumerate((ax1, ax2)):
        text = r'\textbf{{{letter}}}'.format(letter=string.ascii_uppercase[n])
        ax.text(-0.08, 1.04, text, transform=ax.transAxes,
                weight='semibold', size=24)
    plt.tight_layout()
    plt.savefig(f'supp_6.svg', bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    supp_4()
    supp_5()
    supp_6()
