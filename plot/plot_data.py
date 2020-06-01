import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec
from matplotlib import rc
from matplotlib.ticker import ScalarFormatter, FuncFormatter

from utils.correlation import multivariate_gaussian

rc('text', usetex=True)

benchmarks = ['dicarlo.Majaj2015.V4-pls', 'dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n',
              'fei-fei.Deng2009-top1']

benchmarks_small = ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n']

my_palette = ['#168A82', '#FF3210', '#1D9FB3', '#995d13']
my_palette_light = ['#75FF93', '#B3F5FF', '#FFBAAD', '#cfa256']
red_palette = ['#FF3210', '#803E33', '#CC290C', '#FF7D66', '#99665D', '#424949']
green_palette = ['#1DB33D', '#75FF93', '#29FF57', '#58BF6E', '#147F2C']
blue_palette = ['#168A82', '#5D7575', '#9FE0DC', '#10635E', '#709D9A', '#2B3D3C']
# blue_palette = ['#168A82', '#5D7575', '#259C9C', '#36E3E3', '#9AC3C3']
grey_palette = ['#ACB9C6', '#ABB2B9', '#7C8287', '#6C7175', '#24303B']
my_palette_mix = ['#995d13', '#cfa256', '#0c7169', '#58b0a7', '#296e85', '#549ebe', ]
combi = ['#1D9FB3', '#FFBAAD', '#FF3210', '#B3F5FF', '#1DB31E', '#91FF91', '#995d13', '#cfa256']
pal_of_pals = [red_palette, blue_palette, green_palette, grey_palette]


def get_all_perturbations():
    return ['', '_random', '_jumbler', '_kernel_jumbler', '_channel_jumbler', '_norm_dist', '_norm_dist_kernel']


def get_all_models():
    return ['CORnet-S', 'alexnet', 'resnet101']


def formatter(x, pos, percent=False, million=False):
    if million:
        mil = (x / 1000000)
        if mil >= 1:
            x = f'{int(mil)}'
        else:
            x = ('%.4f' % mil).rstrip('0').lstrip('0')
        return x
    if x != 0:
        if abs(x) > 1:  # hacky
            x = f'{int(x)}'
        else:
            x = ('%.2f' % x).lstrip('0').rstrip('0').rstrip('.')
    else:
        x = '0'
    if percent:
        x = f'{x}\%'
    return x


def output_paper_quality(ax, title=None, xlabel=None, ylabel=None, percent=False, percent_x=False, millions=False):
    ax.set_title(title, weight='semibold', size=20)
    ax.set_xlabel(xlabel, size=18)  # weight='semibold',
    ax.set_ylabel(ylabel, size=18)  # weight='semibold',
    func_percent = lambda x, pos: formatter(x, pos, percent)
    func_mil = lambda x, pos: formatter(x, pos, million=millions)
    if percent_x:
        ax.xaxis.set_major_formatter(FuncFormatter(func_percent))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(func_mil))
    ax.yaxis.set_major_formatter(FuncFormatter(func_percent))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(bottom=False, left=False)

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


def plot_data(benchmarks, data, labels, name, scale_fix=None):
    sns.set()
    sns.set_context("paper")
    x = np.arange(len(benchmarks))
    plt.xticks(x, labels, rotation='vertical', fontsize=8)
    print(data)
    for key, value in data.items():
        if key is 'Full':
            plt.plot(x, value, label=key, linestyle="", marker="o", color='#424949')
        else:
            plt.plot(x, value, label=key, linestyle="", marker="o")
    plt.legend()
    plt.tight_layout()
    if scale_fix:
        plt.ylim(scale_fix[0], scale_fix[1])
    plt.savefig(f'{name}.png')
    plt.show()


def plot_data_base(data, name, x_name='', y_name='', x_values=None, x_labels=None, x_ticks=None, scale_fix=None,
                   rotate=False, alpha=1.0,
                   million=False,
                   log=False, use_xticks=False, percent=False, special_xaxis=False, annotate=False,
                   annotate_pos=0, ax=None, only_blue=False, legend=True, palette=None):
    sns.set()
    sns.set_context("talk")
    sns.set_style("white")
    show = True
    if ax is None:
        plt.figure(figsize=(15, 10))
        ax = plt.gca()
    else:
        show = False
    if x_values is None and x_labels is None:
        x_values = np.arange(len(list(data.values())[0]))
    if x_values is None:
        x_values = x_labels
    if x_ticks is None:
        x_ticks = x_labels
    if rotate:
        ax.xticks(rotation='vertical', fontsize=12)
    index = 0
    palette = palette[:len(data) - 1] + ['#ABB2B9']
    if only_blue:
        cols = sns.color_palette("PuBuGn_d", len(data) - 1)
        cols.reverse()
        palette = cols + ['#424949']
    elif palette is None:
        palette = grey_palette
    for key, value in data.items():
        if 'Not corrected' in key:
            ax.plot(x_labels[-1], data[key][-1], label=key, linestyle="", marker="o", alpha=alpha, color=palette[index])
            index += 1
            lab = x_values[-1]
        elif use_xticks and key in ['base', 'base_untrained', 'base_trained', 'Mean']:
            ax.plot(x_values, data[key], label=key, linestyle="solid", marker=".", alpha=1, color='black')
            col = 'black'
            index += 1
            alpha = 1
            lab = x_values
        elif key is 'Full' or key == 'Standard train' or key == 'Standard training' and special_xaxis:
            ax.plot(x_values[key], data[key], label=key, scalex=True, linestyle="dashed", marker=".", alpha=alpha,
                    color='#424949')
            col = '#424949'
            lab = x_values
        elif key in ['base', 'base_untrained', 'base_trained', 'Mean']:
            ax.plot(x_values, data[key], label=key, scalex=True, linestyle="solid", marker=".", alpha=1,
                    color='black')
            col = 'black'
            alpha = 1
            lab = x_values
        elif special_xaxis:
            labels = x_values[key]
            ax.plot(labels, data[key], label=key, scalex=True, linestyle="dashed", marker=".", alpha=alpha,
                    color=palette[index])
            col = palette[index]
            lab = labels
            index += 1
        else:
            ax.plot(x_values, data[key], label=key, scalex=True, linestyle="-", marker=".", alpha=alpha,
                    color=palette[index])
            col = palette[index]
            lab = x_values
            index += 1
        if annotate:
            ax.annotate(key,  # this is the text
                        (lab[annotate_pos], data[key][annotate_pos]),  # this is the point to label
                        textcoords="offset points",  # how to position the text
                        xytext=(-10, 19), alpha=alpha,  # distance from text to points (x,y)
                        ha='center', color=col)
    if log:
        ax.set_xscale('symlog')
        ax.xaxis.set_major_formatter(ScalarFormatter())
    if percent:
        ax.set_yticks(range(0, 110, 20))
        # ax.set_xticklabels(x_labels)
    if legend:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), frameon=False, ncol=min(len(data), 3))
    if scale_fix:
        ax.ylim(scale_fix[0], scale_fix[1])

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    output_paper_quality(ax, name, x_name, y_name, percent, millions=million)
    ax.tick_params(bottom=True, left=False)

    if show:
        plt.tight_layout()
        file_name = name.replace(' ', '_')
        plt.savefig(f'{file_name}.svg')
        sns.despine()
        plt.show()


def plot_data_double(data, data2, name, err=None, err2=None, x_labels=None, x_name='', y_name='', scale_fix=None,
                     rotate=False, alpha=1.0,
                     x_ticks=None, log=False, x_ticks_2=None, percent=False, annotate_pos=None, data_labels=None,
                     ylim=None, scatter=False,
                     ax=None, percent_x=False, pal=my_palette + my_palette_light, gs=None, million=False):
    sns.set()
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    show = False
    if gs is not None:
        ax = plt.subplot(gs)
        show = False
    elif ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        show = True
    if x_ticks is None:
        x_ticks = np.arange(len(list(data.values())[0]))
    if rotate:
        ax.xticks(rotation='vertical', fontsize=12)
    if data2 and len(data2['Score']) > 0:
        if err2:
            ax.errorbar(x_ticks_2, data2['Score'], yerr=err2['Score'],
                        label='Kaiming Normal + Downstream Training (KN+DT)',
                        linestyle="--",
                        marker="^", alpha=alpha, color='#5d7575', )
        else:
            ax.plot(x_ticks_2, data2['Score'], label='Kaiming Normal + Downstream Training (KN+DT)', scalex=True,
                    linestyle="--",
                    marker="^", alpha=alpha, color='#5d7575', )
        full_y = data2['Score'][0]
        full_x = x_ticks_2[0]
        if data_labels:
            ax.annotate(s='Standard Training', xy=(full_x, full_y), color='#5d7575')
            ax.annotate(s='Downstream Training', xy=(x_ticks_2[-1], data2['Score'][-1]), color='#5d7575')
    idx = 0
    legend = False
    for key, value in data.items():
        if isinstance(pal[idx], list):
            cols = pal[idx][:len(x_ticks[key])]
        else:
            cols = pal[idx]
        if err:
            ax.errorbar(x_ticks[key], data[key], yerr=err[key], label=key, linestyle="--", marker="o", alpha=alpha,
                        color=cols)
        elif scatter:
            ax.plot(x_ticks[key], data[key], label=key, scalex=True, linestyle='',
                    marker="o", alpha=0.9, color=cols, )
        else:
            ax.plot(x_ticks[key], data[key], label=key, scalex=True, linestyle="--", marker="o", alpha=alpha,
                    color=cols)
        y_val = data[key]
        if data_labels is not None and isinstance(data_labels, dict):
            for x, label, y, id in zip(x_ticks[key], data_labels[key], y_val, range(len(y_val))):
                ax.annotate(label,  # this is the text
                            (x, y),  # this is the point to label
                            textcoords="offset points",  # how to position the text
                            xytext=(0, 10),  # distance from text to points (x,y)
                            ha='center', color=cols[id] if isinstance(cols, list) else cols)  # color='#ABB2B9',
        elif annotate_pos is not None:
            label = data_labels[idx] if isinstance(data_labels, list) else key
            if len(x_ticks[key]) > 0:
                ax.annotate(label,  # this is the text
                            (x_ticks[key][annotate_pos], data[key][annotate_pos]),  # this is the point to label
                            textcoords="offset points",  # how to position the text
                            xytext=(-10, 19),  # distance from text to points (x,y)
                            ha='center', color=cols[0] if isinstance(cols, list) else cols)
        idx += 1
    if log:
        ax.set_xscale('symlog')
    if ylim is not None:
        ax.set_ylim(ylim)
    if x_labels is not None:
        ax.set_xticks(x_labels)
    if percent:
        ax.set_yticks(range(0, 110, 20))
    rows = 1
    rows = 2 if len(data) >= 4 else rows
    rows = 3 if len(data) >= 6 else rows
    if annotate_pos is None:  # or isinstance(data_labels, list):
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), frameon=False, ncol=int(rows))
    if scale_fix:
        ax.ylim(scale_fix[0], scale_fix[1])
    output_paper_quality(ax, name, x_name, y_name, percent, percent_x=percent_x, millions=million)
    if show:
        file_name = name.replace(' ', '_')
        plt.savefig(f'{file_name}.svg')
        plt.show()


def plot_two_scales(data, name, x_labels=None, x_name='', y_name='', y_name2='', scale_fix=None, rotate=False,
                    alpha=1.0,
                    base_line=0):
    sns.set()
    sns.set_context("paper")
    if rotate:
        plt.xticks(rotation='vertical', fontsize=8)
    print(data)
    isFirst = True
    fig, ax1 = plt.subplots()
    for key, value in data.items():
        if x_labels is None:
            x_labels = np.arange(len(value))
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


def plot_date_map_custom_x(data, name, label_field='layer', x_name='', y_name='', scale_fix=None):
    sns.set()
    sns.set_context("talk")
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 10))
    x = data[label_field]
    print(data)
    for key, value in data.items():
        if key is not label_field:
            length = len(data[key])
            plt.plot(x[0:length], data[key], label=key, linestyle='-', marker=".")
    plt.plot([x[-1]], [0.5], label='Test', marker='o')
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
    sns.set()
    sns.set_context("paper")
    x = np.arange(len(data[label_field]))
    # data['layer']
    # plt.xticks(x, x,rotation='vertical', fontsize=8)
    print(data)
    for key, value in data.items():
        if key is not label_field:
            plt.plot(data[label_field], data[key], label=key, linestyle="", marker="o")
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
        ax.set_title(f'Parameter {param}', pad=3)
        entries, bin_edges, patches = ax.hist(set, alpha=0.5, bins=bins, normed=True, range=range, fill=True,
                                              histtype='bar')
        xt = plt.xticks()[0]
        xmin, xmax = min(xt), max(xt)
        lnspc = np.linspace(xmin, xmax, len(set))
        pos[idx] = lnspc
        print(set)
        ax.legend(prop={'size': 5})
        axes.append(ax)
        idx += 1
    mult = multivariate_gaussian(data)
    results = mult.pdf(pos)
    for i in range(len(axes)):
        ax = axes[i]
        plt.plot(pos[i], results[i])
    plt.tight_layout()
    file_name = name.replace(' ', '_')
    plt.savefig(f'{file_name}.png')
    plt.show()


def plot_heatmap(data, col_labels, row_labels, title, ax=None, percent=False, **kwargs):
    show = False
    if ax is None:
        show = True
        sns.set()
        sns.set_context("paper")
        plt.figure(figsize=(15, 15))
    ax = sns.heatmap(data, ax=ax, **kwargs)
    ax.set_yticks(np.arange(len(kwargs['yticklabels'])) + 0.5)
    ax.set_xticks(np.arange(len(kwargs['xticklabels'])) + 0.5)
    ax.set_title(title, size=20)
    ax.set_xlabel(col_labels, weight='semibold', size=18)
    ax.set_ylabel(row_labels, weight='semibold', size=18)
    ax.xaxis.set_tick_params(rotation=45)
    ax.yaxis.set_tick_params(rotation=45)
    if show:
        file_name = title.replace(' ', '_')
        plt.savefig(f'{file_name}.png')
        plt.show()


def plot_bar_benchmarks(data, labels, title='', y_label='', file_name='bar_plots', yerr=None, line=None, label=False,
                        grey=True,
                        gs=None, ax=None):
    sns.set()
    sns.set_style("white", {'grid.color': '.95', })
    sns.set_context("talk")
    # sns.set_context("paper")
    show = False
    if gs is None and ax is None:
        fig, ax = plt.subplots(figsize=(15, 10), gridspec_kw={'bottom': 0.17})
        show = True
    elif ax is None:
        ax = plt.subplot(gs)
    bars = len(data)
    step_size = int(bars / 5) + 1
    step_size = 1.5
    x = np.arange(0, step_size * len(labels), step_size)
    if bars < 3:
        width = step_size / 0.4
        font_size = 20
        # bars = 1
    elif bars < 5:
        width = step_size / 8  # the width of the bars
        font_size = 26
    else:
        width = step_size / 10
        font_size = 20
    left_edge = ((bars / 2) * width)

    idx = 0
    axes = []
    if grey:
        if len(data) > len(combi) + 2:
            pal = ['#ABB2B9'] + sns.color_palette("muted", len(data) - 2) + ['#424949']
        else:
            if len(data) - 2 > 4:
                pal = ['#ABB2B9'] + combi[:(len(data) - 2)] + ['#424949']
            else:
                pal = ['#ABB2B9'] + my_palette[:(len(data) - 2)] + ['#424949']
    else:
        pals = [blue_palette, '#ABB2B9']
        pals = [blue_palette[1], blue_palette[0], '#ABB2B9']

    for key, value in data.items():
        if label:
            colors = pals[idx]
            if yerr is not None:
                rects = ax.bar(x - left_edge + (idx * width), value, width, yerr=yerr[key], label='', color=colors)
            else:
                rects = ax.bar(x - left_edge + (idx * width), value, width, label='', color=colors)
            axes.append(rects)
            for rect in rects:
                height = rect.get_height()
                if height < 0.1:
                    ax.annotate(key,
                                xy=(rect.get_x() + rect.get_width() / 2, height + 0.01),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points", rotation='vertical',
                                ha='center', va='bottom', size=font_size)
                else:
                    ax.annotate(key,
                                xy=(rect.get_x() + rect.get_width() / 2, height - 0.019),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points", rotation='vertical',
                                ha='center', va='top', size=font_size)
        else:
            axes.append(ax.bar(x - left_edge + (idx * width), value, width, label=key, color=pal[idx]))
        idx += 1
    if line:
        ax.axhline(y=line, linewidth=1, color='#424949', linestyle="dashed")
        ax.annotate(r'\textbf{Standard Training}',
                    xy=(-2, line),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points", size=font_size,
                    ha='center', va='bottom')
    ax.set_ylabel(y_label, size=font_size)

    ax.set_xticks(x)
    # ax.set(ylim=[0.0, 0.45])
    ax.legend(loc='lower left', bbox_to_anchor=(-0.2, 1), frameon=False, ncol=len(data))

    for rect in ax.patches:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, '{}'.format(height),
                ha='center', va='bottom')
    output_paper_quality(ax, title, xlabel='', ylabel=y_label)
    ax.set_xticklabels(labels, size=font_size)
    if show:
        plt.tight_layout()
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


def scatter_plot(x, y, y_2=None, x_label=None, y_label=None, labels=None, title=None, trendline=False, scale_fix=None,
                 percent=False, ax=None):
    sns.set()
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    show = False
    if ax is None:
        plt.figure(figsize=(10, 10))
        show = True
        ax = plt.gca()
    corr = np.corrcoef(x, y)[0, 1]
    if title:
        # ax.title(title)
        file_name = title.replace(' ', '_')
    else:
        ax.set_title(f'Correlation: {corr}')
        file_name = f'Correlation_{corr}'
    # plt.xlabel(x_label)
    # plt.ylabel(y_label)
    # colors = ['#ABB2B9'] + (len(x)-3) * ["royalblue"] + 2* ['#ABB2B9']
    # colors = ['#ABB2B9'] + (len(x) - 5) * ["royalblue"] + 4 * ['#ABB2B9']
    # plt.scatter(x, y, color=colors)
    ax.scatter(x, y, color=combi[0])
    if y_2:
        ax.scatter(x, y_2, color='#424949', marker="^", )
    if trendline:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--")
    if scale_fix:
        plt.ylim(scale_fix[0], scale_fix[1])
    if labels:
        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]))
    output_paper_quality(ax, title, x_label, y_label, percent)
    if show:
        plt.savefig(f'{file_name}.png')
        plt.show()
    return corr

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


def plot_matrixImage(matrix, title, size=3):
    # sns.set()
    # sns.set_context("talk")
    fig, ax = plt.subplots(figsize=(20, 20))
    # loc = plticker.MultipleLocator(base=size)
    # ax.xaxis.set_major_locator(loc)
    # ax.yaxis.set_major_locator(loc)

    ax.set_xlabel('Kernels')
    ax.set_ylabel('Filters')
    # ax.xaxis.set_ticks(np.arange(0, matrix.shape[0]), size)
    # ax.yaxis.set_ticks(np.arange(0, matrix.shape[1]), size)
    # ax.grid(which='major', axis='both', linestyle='-')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    # extent = (0, matrix.shape[1], matrix.shape[0], 0)
    plt.imshow(matrix, cmap='gray')  # , extent=extent
    plt.savefig(f'{title}.png')
    plt.show()


def plot_pie(sizes, labels, explode=None):
    sns.set()
    sns.set_context("paper")
    # explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()


if __name__ == '__main__':
    my_palette = []
    my_palette_light = []
    sns.set()
    # palette = sns.color_palette("RdBu_r", 7)
    # my_palette.append(palette[0])
    # my_palette_light.append(palette[1])
    # palette = sns.diverging_palette(255, 133, l=60, n=7, center="dark")
    # my_palette.append(palette[1])
    # my_palette_light.append(palette[0])
    print(sns.color_palette("PuBuGn_d", 8).as_hex())
    # sns.palplot(sns.color_palette("PuBuGn_d", 8))
    both = []
    palette0 = sns.color_palette("RdBu_r", 7)
    palette1 = sns.diverging_palette(220, 20, n=7)
    print(palette1.as_hex())
    print(palette1)
    print(palette0.as_hex())
    print(palette0)
    palette = sns.color_palette("BrBG", 7)
    print(palette.as_hex())
    print(palette)
    my_palette.append(palette1[6])
    both.append(palette1[6])
    my_palette_light.append(palette1[5])
    both.append(palette1[5])
    my_palette.append(palette[6])
    both.append(palette[6])
    my_palette_light.append(palette[5])
    both.append(palette[5])
    my_palette.append(palette[0])
    both.append(palette[0])
    my_palette_light.append(palette[1])
    both.append(palette[1])
    my_palette.append(palette0[0])
    both.append(palette0[0])
    my_palette_light.append(palette0[1])
    both.append(palette0[1])
    # print(both)
    sns.palplot(my_palette)
    full = ['#ABB2B9'] + my_palette + ['#424949']
    sns.palplot(full)
    plt.show()
    full = ['#ABB2B9'] + my_palette_light + ['#424949']
    sns.palplot(full)
    plt.show()
    sns.palplot(both)
    # print(both.as_hex())
    plt.show()
    # print(my_palette)
    # print(my_palette_light)
