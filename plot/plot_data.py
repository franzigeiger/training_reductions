import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec
from matplotlib.ticker import ScalarFormatter, FuncFormatter

from benchmark.database import load_scores, get_connection
from utils.correlation import multivariate_gaussian

benchmarks = ['dicarlo.Majaj2015.V4-pls', 'dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n',
              'fei-fei.Deng2009-top1']

benchmarks_small = ['dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n']

# my_palette = [(0.24715576253545807, 0.49918708160096675, 0.5765599057376697), (0.7634747047461135, 0.3348456555528834, 0.225892295531744), (0.6000000000000001, 0.3656286043829296, 0.07420222991157246)]
# my_palette_light = [(0.48280569028069864, 0.658420419636426, 0.7123335905099379), (0.837219841163406, 0.5415589592826877, 0.4664048217636615), (0.8129950019223375, 0.635832372164552, 0.33640907343329485)]
my_palette = [(0.24715576253545807, 0.49918708160096675, 0.5765599057376697),
              (0.7634747047461135, 0.3348456555528834, 0.225892295531744), (0.0479046520569012, 0.44144559784698195,
                                                                            0.4100730488273741)]  # (0.6000000000000001, 0.3656286043829296, 0.07420222991157246),
my_palette_light = [(0.48280569028069864, 0.658420419636426, 0.7123335905099379),
                    (0.837219841163406, 0.5415589592826877, 0.4664048217636615), (0.346251441753172, 0.6918108419838525,
                                                                                  0.653056516724337)]  # (0.8129950019223375, 0.635832372164552, 0.33640907343329485),
combi = [(0.7634747047461135, 0.3348456555528834, 0.225892295531744),
         (0.837219841163406, 0.5415589592826877, 0.4664048217636615), '#296e85', '#549ebe',
         (0.0479046520569012, 0.44144559784698195, 0.4100730488273741),
         (0.346251441753172, 0.6918108419838525, 0.653056516724337),
         (0.6000000000000001, 0.3656286043829296, 0.07420222991157246),
         (0.8129950019223375, 0.635832372164552, 0.33640907343329485)]


def get_all_perturbations():
    return ['', '_random', '_jumbler', '_kernel_jumbler', '_channel_jumbler', '_norm_dist', '_norm_dist_kernel']


def get_all_models():
    return ['CORnet-S', 'alexnet', 'resnet101']


def formatter(x, pos, percent=False):
    if x != 0:
        if abs(x) > 1:  # hacky
            x = f'{int(x)}'
        else:
            x = ('%.2f' % x).lstrip('0').rstrip('0')
    else:
        x = '0'
    if percent:
        x = f'{x}%'
    return x


def output_paper_quality(ax, title=None, xlabel=None, ylabel=None, percent=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel, weight='semibold', size=16)
    ax.set_ylabel(ylabel, weight='semibold', size=16)
    func = lambda x, pos: formatter(x, pos, percent)
    ax.xaxis.set_major_formatter(FuncFormatter(formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(func))
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
        if key is 'Full':
            plt.plot(x, value, label=key, linestyle="", marker="o", color='#424949')
        else:
            plt.plot(x, value, label=key, linestyle="", marker="o")
    plt.legend()
    plt.tight_layout()
    if scale_fix:
        plt.ylim(scale_fix[0], scale_fix[1])
    # res.save('test.png')
    plt.savefig(f'{name}.png')
    plt.show()


def plot_data_base(data, name, x_labels=None, x_name='', y_name='', scale_fix=None, rotate=False, alpha=1.0,
                   x_ticks=None, log=False, use_xticks=False, percent=False, special_xaxis=False, annotate=False,
                   annotate_pos=0, ax=None):
    sns.set()
    sns.set_context("talk")
    sns.set_style("white")
    show = True
    if ax is None:
        plt.figure(figsize=(15, 10))
        ax = plt.gca()
    else:
        show = False
    if x_labels is None:
        x_labels = np.arange(len(list(data.values())[0]))
    if rotate:
        ax.xticks(rotation='vertical', fontsize=12)
    print(data)
    # if base_line is not 0:
    #     plt.hlines(base_line, xmin=0, xmax=1, colors='b')
    index = 0
    palette = combi
    # cols = sns.color_palette("PuBuGn_d", len(data)-1)
    # cols.reverse()
    # palette = cols  + ['#ABB2B9']
    for key, value in data.items():
        if 'Not corrected' in key:
            ax.plot(x_labels[-1], data[key][-1], label=key, linestyle="", marker="o", alpha=alpha, color=palette[index])
            index += 1
        elif key in ['base', 'base_untrained', 'base_trained']:
            ax.plot(x_labels, data[key], label=key, linestyle="solid", marker="", alpha=alpha, color=palette[index])
            index += 1
        elif key is 'Full' or key == 'Standard train' or key == 'Standard training' and special_xaxis:
            ax.plot(x_labels[key], data[key], label=key, scalex=True, linestyle="dashed", marker=".", alpha=alpha,
                    color='#424949')
        elif key is 'Full' or key == 'Standard train' or key == 'Standard training':
            ax.plot(x_labels, data[key], label=key, scalex=True, linestyle="dashed", marker=".", alpha=alpha,
                    color='#424949')
        elif special_xaxis:
            labels = x_labels[key]
            ax.plot(labels, data[key], label=key, scalex=True, linestyle="dashed", marker=".", alpha=alpha,
                    color=palette[index])
            index += 1
            y_val = data[key]
        elif use_xticks:
            ax.plot(x_ticks, data[key], label=key, scalex=True, linestyle="-", marker=".", alpha=alpha,
                    color=palette[index])
            index += 1
            y_val = data[key]
        else:
            ax.plot(x_labels, data[key], label=key, scalex=True, linestyle="-", marker=".", alpha=alpha,
                    color=palette[index])
            if annotate:
                ax.annotate(key,  # this is the text
                            (x_labels[annotate_pos], data[key][annotate_pos]),  # this is the point to label
                            textcoords="offset points",  # how to position the text
                            xytext=(-10, 19),  # distance from text to points (x,y)
                            ha='center')
            y_val = data[key]
            index += 1
    if log:
        ax.xscale('symlog')
        ax.xaxis.set_major_formatter(ScalarFormatter())
    if x_ticks:
        ax.xticks(x_ticks)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), frameon=False, ncol=len(data))
    if scale_fix:
        ax.ylim(scale_fix[0], scale_fix[1])
    ax.tick_params(axis='both', which='major', labelsize=16)
    output_paper_quality(ax, name, x_name, y_name, percent)
    ax.tight_layout()
    if show:
        file_name = name.replace(' ', '_')
        plt.savefig(f'{file_name}.png')
        sns.despine()
        plt.show()


def plot_data_double(data, data2, name, x_labels=None, x_name='', y_name='', scale_fix=None, rotate=False, alpha=1.0,
                     x_ticks=None, log=False, x_ticks_2=None, percent=False, exponential=False, data_labels=None,
                     ax=None):
    sns.set()
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    show = False
    if ax is None:
        # sns.set_style("whitegrid")
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        show = True
    if x_labels is None:
        x_labels = np.arange(len(list(data.values())[0]))
    if rotate:
        ax.xticks(rotation='vertical', fontsize=12)
    pal = my_palette_light[:len(data) + 1]
    pal = combi
    idx = 0
    for key, value in data.items():
        ax.plot(x_ticks[key], data[key], label=key, scalex=True, linestyle="--", marker="o", alpha=alpha,
                color=pal[idx])
        y_val = data[key]
        if data_labels:
            for x, label, y in zip(x_ticks[key], data_labels[key], y_val):
                ax.annotate(label,  # this is the text
                            (x, y),  # this is the point to label
                            textcoords="offset points",  # how to position the text
                            xytext=(0, 10),  # distance from text to points (x,y)
                            ha='center', color='#ABB2B9')
        idx += 2
    if data2:
        ax.plot(x_ticks_2, data2['Score'], label='Standard init(Kaiming normal)', scalex=True, linestyle="--",
                marker="^", alpha=alpha, color='#424949', )
        full_y = data2['Score'][0]
        full_x = x_ticks_2[0]
        if data_labels:
            ax.annotate(s='Standard trained', xy=(full_x, full_y), color='#ABB2B9')
    if log:
        ax.set_xscale('symlog')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), frameon=False, ncol=2)
    if scale_fix:
        ax.ylim(scale_fix[0], scale_fix[1])
    output_paper_quality(ax, name, x_name, y_name, percent)
    if show:
        file_name = name.replace(' ', '_')
        plt.savefig(f'{file_name}.png')
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
    plt.figure(figsize=(15, 15))
    ax = sns.heatmap(data, cmap="YlGnBu", center=0, **kwargs)

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
    sns.set_style("white")
    sns.set_context("talk")
    # sns.set_context("paper")
    bars = len(data)
    step_size = int(bars / 5) + 1
    x = np.arange(0, step_size * len(labels), step_size)  # the label locations
    if bars < 5:
        width = step_size / 6  # the width of the bars
        font_size = 26
    else:
        width = step_size / 10
        font_size = 20
    left_edge = ((bars / 2) * width)

    if bars > 8:
        fig, ax = plt.subplots(figsize=(15, 10))
    else:
        fig, ax = plt.subplots(figsize=(15, 10))
    idx = 0
    axes = []
    if len(data) > len(combi) + 2:
        pal = ['#ABB2B9'] + sns.color_palette("muted", len(data) - 2) + ['#424949']
    else:
        pal = ['#ABB2B9'] + combi[:len(data) - 2] + ['#424949']
    # pal = ['#ABB2B9'] + sns.color_palette("coolwarm", len(data)-2) + ['#ABB2B9']
    for key, value in data.items():
        axes.append(ax.bar(x - left_edge + (idx * width), value, width, label=key, color=pal[idx]))
        idx += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label, fontsize=24)
    # ax.set_yticklabels(ax.get_yticklabels(),fontsize=22)
    plt.yticks(fontsize=24)
    # ax.set_title(title)
    ax.set_xticks(x)
    ax.set(ylim=[0.0, 0.6])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), frameon=False, ncol=len(data))
    # ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.01), prop={'size': font_size}, borderaxespad=0, frameon=False,
    #           ncol=2)

    for rect in ax.patches:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, '{}'.format(height),
                ha='center', va='bottom')
    output_paper_quality(ax, title, xlabel='', ylabel=y_label)
    ax.set_xticklabels(labels, fontsize=24)
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
    palette = sns.color_palette("BrBG", 7)
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
    print(both)
    sns.palplot(my_palette)
    full = ['#ABB2B9'] + my_palette + ['#424949']
    sns.palplot(full)
    plt.show()
    full = ['#ABB2B9'] + my_palette_light + ['#424949']
    sns.palplot(full)
    plt.show()
    sns.palplot(both)
    print(both.as_hex())
    plt.show()
    # print(my_palette)
    # print(my_palette_light)
