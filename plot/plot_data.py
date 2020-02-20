import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter

from benchmark.database import load_scores, create_connection
from utils.correlation import multivariate_gaussian

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
    sns.set()
    sns.set_context("paper")
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
    sns.set()
    sns.set_context("paper")
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
    sns.set()
    sns.set_context("paper")
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


def plot_histogram(data, name, bins=100, labels=[], x_axis='Weight distribution', range=None):
    sns.set()
    sns.set_context("paper")
    plt.hist(data, alpha=0.5, bins=bins, range=range, fill=True, histtype='bar', density=True, label=labels)
    plt.legend(prop={'size': 10})
    plt.gca().set(title=name, xlabel=x_axis)
    plt.tight_layout()
    file_name = name.replace(' ', '_')
    plt.savefig(f'{file_name}.png')
    plt.show()


def plot_subplots_histograms(data, name, bins=100, x_axis='Weight distribution', range=None, bounds=None):
    sns.set()
    sns.set_context("paper")
    row = int(np.sqrt(len(data))) + 1
    plt.figure(figsize=(10, 10))
    idx = 0
    dist_names = ['alpha', 'anglit', 'arcsine', 'beta', 'betaprime', 'bradford', 'burr', 'cauchy', 'chi', 'chi2',
                  'cosine', 'dgamma', 'dweibull', 'erlang', 'expon', 'exponweib', 'exponpow', 'f', 'fatiguelife',
                  'fisk', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l', 'genlogistic', 'genpareto', 'genexpon',
                  'genextreme', 'gausshyper', 'gamma', 'gengamma', 'genhalflogistic', 'gilbrat', 'gompertz', 'gumbel_r',
                  'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'invgamma', 'invgauss',
                  'invweibull', 'johnsonsb', 'johnsonsu', 'ksone', 'kstwobign', 'laplace', 'logistic', 'loggamma',
                  'loglaplace', 'lognorm', 'lomax', 'maxwell', 'mielke', 'nakagami', 'ncx2', 'ncf', 'nct', 'norm',
                  'pareto', 'pearson3', 'powerlaw', 'powerlognorm', 'powernorm', 'rdist', 'reciprocal', 'rayleigh',
                  'rice', 'recipinvgauss', 'semicircular', 't', 'triang', 'truncexpon', 'truncnorm', 'tukeylambda',
                  'uniform', 'vonmises', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy']
    dist_names = ['expon', 'halflogistic', 'norm', 'powernorm', 'genexpon', 'laplace', 'logistic', 'loggamma',
                  'loglaplace']
    axes = []
    param_arrap = np.array(list(data.values()))
    print(param_arrap.shape)
    pos = np.empty(param_arrap.shape[1] + (2,))
    for param, set in data.items():
        ax = plt.subplot(row, row, idx + 1)
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_title(f'Parameter {param}', pad=3)
        entries, bin_edges, patches = ax.hist(set, alpha=0.5, bins=bins, normed=True, range=range, fill=True,
                                              histtype='bar')
        xt = plt.xticks()[0]
        xmin, xmax = min(xt), max(xt)
        lnspc = np.linspace(xmin, xmax, len(set))
        pos[idx] = lnspc
        print(set)

        # m, s = stats.norm.fit(set) # get mean and standard deviation
        # param = params[:,:,i]
        # result = minimize(negLogLikelihood,  # function to minimize
        #                   x0=np.zeros(1),     # start value
        #                   args=(set,),      # additional arguments for function
        #                   method='Powell',   # minimization method, see docs
        #                   options={'maxiter': 20000},
        #                   bounds = (bounds[idx])
        #                   )
        # print(result)
        # pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval
        # plt.plot(lnspc, pdf_g, label="Norm")
        # for dist_name in dist_names:
        #     dist = getattr(scipy.stats, dist_name)
        #     param = dist.fit(set)
        #     pdf_fitted = dist.pdf(lnspc, *param[:-2], loc=param[-2], scale=param[-1])
        #     plt.plot(lnspc, pdf_fitted, label=dist_name)
        #     plt.xlim(xmin, xmax)

        # bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
        # parameters, cov_matrix = curve_fit(poisson, bin_middles, entries)
        # print(f'Parameter: {param}, value{parameters}')
        # plt.plot(lnspc, poisson(lnspc, parameters[0]), label="Norm")
        # ax.plot(set, kde.pdf(set), label='KDE')
        # ax.hist(set, alpha=0.5, bins=bins, range=range, fill=True, histtype='bar', density=True)
        ax.legend(prop={'size': 5})
        axes.append(ax)
        idx += 1
    mult = multivariate_gaussian(data)
    results = mult.pdf(pos)
    for i in range(len(axes)):
        ax = axes[i]
        plt.plot(pos[i], results[i])
    # plt.gca().set(title=name)
    plt.tight_layout()
    file_name = name.replace(' ', '_')
    plt.savefig(f'{file_name}.png')
    plt.show()


def plot_heatmap(data, col_labels, row_labels, title, **kwargs):
    sns.set()
    sns.set_context("paper")

    ax = sns.heatmap(data, linewidths=.5, annot=True, cmap="YlGnBu", center=0, square=True, **kwargs)

    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation='horizontal', fontsize=7)
    plt.yticks(rotation='horizontal', fontsize=7)
    ax.set_title(title)
    plt.tight_layout()
    file_name = title.replace(' ', '_')
    plt.savefig(f'{file_name}.png')
    plt.show()


def plot_bar_benchmarks(data, labels, title='', y_label='', file_name='bar_plots'):
    sns.set()
    sns.set_context("paper")
    plt.figure(figsize=(15, 15))
    bars = len(data)
    step_size = int(bars / 5) + 1
    x = np.arange(0, step_size * len(labels), step_size)  # the label locations
    width = step_size / 10  # the width of the bars
    left_edge = ((bars / 2) * width)

    fig, ax = plt.subplots()
    idx = 0
    axes = []
    for key, value in data.items():
        axes.append(ax.bar(x - left_edge + (idx * width), value, width, label=key))
        idx += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc=1, prop={'size': 5})

    for rect in ax.patches:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, '{}'.format(height),
                ha='center', va='bottom')
    fig.tight_layout()
    plt.savefig(f'{file_name}.png')
    plt.show()


def plot_images(img, size, labels, theta):
    idx = 0
    plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(int(len(img) / size) + 1, size, width_ratios=[1] * size,
                           wspace=0.5, hspace=0.5, top=0.95, bottom=0.05, left=0.1, right=0.95)
    for j in range(1 + int(len(img) / size)):  # in zip(axes, range(weights.shape[0])):
        for i in range(size):
            if idx < len(img):
                ax = plt.subplot(gs[j, i])
                ax.set_title(labels[idx], pad=3)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.title.set_fontsize(14)
                plt.imshow(img[idx], cmap='gray')
                idx += 1

    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    # plt.tight_layout()
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(f'gabors_{theta}.png')
    plt.show(bbox_inches='tight', pad_inches=0)


def plot_3d(x, y, z, name):
    sns.set()
    sns.set_context("paper")
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, cmap='Greens')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(name)
    file_name = name.replace(' ', '_')
    plt.savefig(f'{file_name}.png')
    plt.show()


def plot_2d(x, y, name):
    sns.set()
    sns.set_context("paper")
    plt.plot(x, y, linestyle="", marker=".")
    # plt.xlabel('x')
    # plt.ylabel('y')
    plt.gca().set(title=name)
    file_name = name.replace(' ', '_')
    plt.savefig(f'{file_name}.png')
    plt.show()