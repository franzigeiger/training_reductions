import itertools

import numpy as np

from analysis.time_analysis import benchmarks
from base_models.global_data import best_brain_avg, layer_random, random_scores, \
    convergence_epoch, convergence_images
from benchmark.database import get_connection, load_scores, load_model_parameter, load_error_bared
from plot.plot_data import plot_data_double, blue_palette, grey_palette, green_palette
from runtime.compression import get_params


def plot_performance(imagenet=True, entry_models=[best_brain_avg], all_labels=[], convergence=False, ax=None,
                     selection=[], log=False):
    conn = get_connection()
    names = []
    for model in random_scores.keys():
        if convergence and model in convergence_epoch:
            postfix = f'_epoch_{convergence_epoch[model]:02d}'
        else:
            postfix = f'_epoch_06'
        if model != "CORnet-S_random":
            names.append(f'{model}{postfix}')
        else:
            names.append(model)
    performance = load_model_parameter(conn)
    model_dict = load_scores(conn, names, benchmarks)
    time2 = []
    data2 = {'Score': []}

    for model, layer in random_scores.items():
        if model == "CORnet-S_random":
            postfix = ''
        elif convergence and model in convergence_epoch:
            postfix = f'_epoch_{convergence_epoch[model]:02d}'
        else:
            postfix = f'_epoch_06'
        high = np.mean(model_dict[f'CORnet-S_full_epoch_{convergence_epoch["CORnet-S_full"]}'][selection])
        perc = (np.mean(model_dict[f'{model}{postfix}'][selection]) / high) * 100
        if layer in performance:
            data2['Score'].append(perc)
            time2.append(performance[layer])

    data = {}
    time = {}
    labels = {}
    for entry_model, name in zip(entry_models, all_labels):
        names = []
        for model in entry_model.keys():
            if convergence and model in convergence_epoch:
                postfix = f'_epoch_{convergence_epoch[model]:02d}'
            else:
                postfix = f'_epoch_06'
            names.append(f'{model}{postfix}')
        model_dict = load_scores(conn, names, benchmarks)
        time[name] = []
        data[name] = []
        for model, layer in entry_model.items():
            if convergence and model in convergence_epoch:
                postfix = f'_epoch_{convergence_epoch[model]:02d}'
            else:
                postfix = f'_epoch_06'
            perc = (np.mean(model_dict[f'{model}{postfix}'][selection]) / high) * 100
            if layer in performance:
                data[name].append(perc)
                time[name].append(performance[layer])
        short = name.split('(')[1][:-1]
        labels[name] = [f'{value.split(".")[0]}_{short}' for value in entry_model.values()]

    if imagenet:
        title = f'Imagenet score vs training time'
        y = r'Imagenet performance [% of standard training]'
    else:
        title = f'Brain-Score Benchmark mean(V4, IT, Behavior) vs training time'
        if len(selection) == 3:
            y = r"mean(V4, IT, Behavior) [% of standard training]"
        else:
            y = r"mean(V1,V2,V4,IT,Behavior) [% of standard training]"
    plot_data_double(data, data2, '', x_name='Training time [Milliseconds/Epoch]', x_labels=[], y_name=y, x_ticks=time,
                     x_ticks_2=time2, percent=True, data_labels=labels, ax=ax, log=log)


def plot_num_params_epochs(imagenet=False, entry_models=[], all_labels=[], epochs=[], convergence=False, ax=None,
                           selection=[], log=False, layer_random=layer_random):
    conn = get_connection()
    full = np.mean(get_full(conn, convergence)[selection])
    data2 = {}
    labels = []
    params = {}
    for entry_model, name in itertools.chain([(layer_random, 'Kaiming Normal + Downstream Training (KN+DT)')],
                                             zip(entry_models, all_labels)):
        short = name.split('(')[1][:-1]
        for epoch in epochs:
            name_epoch = f'{short} Epoch {epoch:02d}'
            data2[name_epoch] = []
            params[name_epoch] = []
        data2[f'{short} Convergence'] = []
        params[f'{short} Convergence'] = []
        mod_params = get_model_params(entry_model.keys())
        names = []
        for model in entry_model.keys():
            if model == "CORnet-S_random":
                names.append(model)
            else:
                conv = convergence_epoch[model] if model in convergence_epoch else 100
                for epoch in epochs:
                    if epoch < conv:
                        names.append(f'{model}_epoch_{epoch:02d}')
                if convergence and model in convergence_epoch:
                    names.append(f'{model}_epoch_{conv:02d}')

        model_dict = load_scores(conn, names, benchmarks)
        for model in names:
            epoch = model.split('_')[-1]
            base_model = model.partition('_epoch')[0]
            percent = (np.mean(model_dict[model][selection]) / full) * 100
            if int(epoch) == convergence_epoch[base_model]:
                name_epoch = f'{short} Convergence'
                data2[name_epoch].append(percent)
                params[name_epoch].append(mod_params[base_model])
            if int(epoch) in epochs:
                name_epoch = f'{short} Epoch {epoch}'
                data2[name_epoch].append(percent)
                params[name_epoch].append(mod_params[base_model])
        labels = labels + [f'Epoch {ep}' for ep in epochs] + ['Convergence']

    if imagenet:
        title = f'Imagenet score vs number of parameter'
        y = r'Imagenet performance [% of standard training]'
    else:
        title = f'Brain-Score Benchmark mean(V4, IT, Behavior) vs number of parameter'
        if len(selection) == 3:
            y = r"mean(V4, IT, Behavior) [% of standard training]"
        else:
            y = r"mean(V1,V2,V4,IT,Behavior) [% of standard training]"
    col = grey_palette[:len(epochs) + 1] + blue_palette[:len(epochs) + 1] + green_palette[
                                                                            :len(epochs) + 1] + grey_palette[
                                                                                                :len(epochs) + 1]
    plot_data_double(data2, {}, '', x_name='Number of trained parameters [Million]', x_labels=[], y_name=y,
                     data_labels=labels,
                     x_ticks=params, pal=col, x_ticks_2=[], percent=True, ax=ax, million=True, ylim=[0, 100],
                     annotate_pos=0, log=log, )


def plot_num_params_images(imagenet=False, entry_models=[], all_labels=[], images=[], convergence=False, ax=None,
                           selection=[], log=False, layer_random=layer_random):
    conn = get_connection()
    full = np.mean(get_full(conn, convergence)[selection])
    data2 = {}
    labels = []
    params = {}
    for entry_model, name in itertools.chain([(layer_random, 'Kaiming Normal + Downstream Training (KN+DT)')],
                                             zip(entry_models, all_labels)):
        short = name.split('(')[1][:-1]
        for img in images:
            name_epoch = f'{short} {img} Imgs'
            data2[name_epoch] = []
            params[name_epoch] = []
        data2[f'{short} Full'] = []
        params[f'{short} Full'] = []
        mod_params = get_model_params(entry_model.keys())
        names = []
        for model in entry_model.keys():
            if model == "CORnet-S_random":
                names.append(model)
            else:
                names.append(f'{model}_epoch_{convergence_epoch[model]}')
                model = model.split('_seed42')[0]
                for img in images:
                    model_img = f'{model}_img{img}'
                    conv = convergence_images[model_img] if model_img in convergence_images else 20
                    names.append(f'{model_img}_epoch_{conv:02d}')

        model_dict = load_scores(conn, names, benchmarks)
        for model in names:
            percent = (np.mean(model_dict[model][selection]) / full) * 100
            if 'img' not in model:
                name_epoch = f'{short} Full'
                base_model = model.partition('_epoch')[0]
                data2[name_epoch].append(percent)
                params[name_epoch].append(mod_params[base_model])
            else:
                img = model.split('_')[-3].partition('g')[2]
                base_model = model.partition('_img')[0]
                name_epoch = f'{short} {img} Imgs'
                data2[name_epoch].append(percent)
                params[name_epoch].append(mod_params[base_model])
            labels = labels + [f'{ep} Images' for ep in images] + ['Convergence']

    if imagenet:
        title = f'Imagenet score vs number of parameter'
        y = r'Imagenet performance [% of standard training]'
    else:
        title = f'Brain-Score Benchmark mean(V4, IT, Behavior) vs number of parameter'
        if len(selection) == 3:
            y = r"mean(V4, IT, Behavior) [% of standard training]"
        else:
            y = r"mean(V1,V2,V4,IT,Behavior) [% of standard training]"
    col = grey_palette[:len(images) + 1] + blue_palette[:len(images) + 1] + green_palette[
                                                                            :len(images) + 1] + grey_palette[
                                                                                                :len(images) + 1]
    plot_data_double(data2, {}, '', x_name='Number of trained parameters [Million]', x_labels=[], y_name=y,
                     x_ticks=params, pal=col, data_labels=labels, ylim=[0, 100],
                     x_ticks_2=[], percent=True, ax=ax, million=True, annotate_pos=0, log=log)


def get_model_params(models, hyperparams=True):
    mod_params = {}
    for model in models:
        model_og = model
        if model.endswith('BF'):
            model = model.replace('_BF', '')
        if model.endswith('seed42'):
            model = model.replace('_seed42', '')
        if model == "CORnet-S_random":
            mod_params[model_og] = 0
        else:
            params = get_params(model, hyperparams)
            mod_params[model_og] = params
            mod_params[model] = params
    return mod_params


def plot_num_params(imagenet=False, entry_models=[], all_labels=[], convergence=False, ax=None, selection=[], log=False,
                    layer_random=layer_random, pal=None, percent=True, ylim=None):
    conn = get_connection()
    full_score = get_full(conn, convergence)
    full = np.mean(full_score[selection])
    names = []
    data = {'Score': []}
    err = {'Score': []}
    model_dict = load_error_bared(conn, layer_random.keys(), benchmarks, convergence=convergence)

    for model, layer in layer_random.items():
        if percent:
            frac = (np.mean(model_dict[model][selection]) / full) * 100
            percent_err = (np.mean(model_dict[model][6:][selection]) / full) * 100
        else:
            frac = np.mean(model_dict[model][selection])
            percent_err = np.mean(model_dict[model][6:][selection])
        print(f'MOdel {model} has score {frac}')
        data['Score'].append(frac)
        err['Score'].append(percent_err)
    data2 = {}
    err2 = {}
    labels = {}
    params = {}
    for entry_model, name in zip(entry_models, all_labels):
        data2[name] = []
        params[name] = []
        err2[name] = []
        for model in entry_model.keys():
            print(model)
            if model.endswith('BF'):
                model = model.replace('_BF', '')
            if model.endswith('seed42'):
                model = model.replace('_seed42', '')
            if model == "CORnet-S_random":
                params[name].append(0)
            else:
                params[name].append(get_params(model))

        model_dict = load_error_bared(conn, entry_model.keys(), benchmarks)
        for model in entry_model.keys():
            if percent:
                frac = (np.mean(model_dict[model][selection]) / full) * 100
                percent_err = (np.mean(model_dict[model][6:][selection]) / full) * 100
            else:
                frac = np.mean(model_dict[model][selection])
                percent_err = np.mean(model_dict[model][6:][selection])
            print(f'MOdel {model} has score {frac}')
            data2[name].append(frac)
            err2[name].append(percent_err)
        if '(' in name:
            short = name.split('(')[1][:-1]
        else:
            short = name
        if len(entry_model) == 1:
            labels[name] = [short, f'{short} Transfer']
        else:
            labels[name] = [short]

    params2 = []
    for model in layer_random.keys():
        if model.endswith('BF'):
            model = model.replace('_BF', '')
        if model == "CORnet-S_random":
            params2.append(0)
        else:
            params2.append(get_params(model))
    if imagenet:
        title = f'Imagenet score vs number of parameter'
        y = r'\textbf{Imagenet performance}'
    else:
        title = f'Brain-Score Benchmark mean(V4, IT, Behavior) vs number of parameter'
        if len(selection) == 3:
            y = r"\textbf{mean(V4, IT, Behavior)}"
        else:
            y = r"\textbf{mean(V1,V2,V4,IT,Behavior)}"
    if percent:
        y = f'{y} [\% of standard training]'

    if pal is None:
        pal = blue_palette
    plot_data_double(data2, data, '', err=err2, err2=err, x_name=r'\textbf{Number of trained parameters} [Million]',
                     x_labels=None, ylim=ylim,
                     y_name=y, x_ticks=params, pal=pal, percent=percent,
                     x_ticks_2=params2, data_labels=labels, ax=ax, million=True, log=log, annotate_pos=0)


def image_epoch_score(models, imgs, epochs, selection=[], axes=None, percent=True):
    names = []
    conn = get_connection()
    params = {}
    data = {}
    for model, label in models.items():
        data[label] = []
        params[label] = []
        if model == 'CORnet-S_cluster2_v2_IT_trconv3_bi':
            model1 = model
        else:
            model1 = model
        for img in imgs:
            name = f'{model}_img{img}'
            for epoch in epochs:
                if epoch % 1 == 0:
                    names.append(f'{name}_epoch_{epoch:02d}')
                else:
                    names.append(f'{name}_epoch_{epoch:.1f}')
            if name in convergence_images:
                names.append(f'{name}_epoch_{convergence_images[name]}')
        names.append(f'{model1}_epoch_{convergence_epoch[model1]:02d}')
    parameter = get_model_params(models, False)
    names.append('CORnet-S_full_epoch_43')
    model_dict = load_scores(conn, names, benchmarks)
    full = np.mean(model_dict['CORnet-S_full_epoch_43'][selection])
    high_x = 0
    high_y = 0
    val = 0
    for model in names:
        if percent:
            frac = (np.mean(model_dict[model][selection]) / full) * 100
        else:
            frac = np.mean(model_dict[model][selection])
        if frac > 0.0:
            if 'img' not in model:
                base_model = model.partition('_epoch')[0]
                epoch = float(model.partition('_epoch_')[2])
                data[models[base_model]].append(frac)
                score = (1280000 * epoch * (parameter[base_model] / 1000000))  #
                print(f'Model {base_model} in epoch {epoch} with full imagenet set '
                      f'leads to score {score} with brain score {frac}')
                params[models[base_model]].append(score)
            else:
                base_model = model.partition('_img')[0]
                imgs = int(model.partition('_img')[2].partition('_')[0])
                epoch = float(model.partition('_img')[2].partition('_epoch_')[2])
                score = (imgs * epoch * (parameter[base_model] / 1000000))  # (parameter[base_model] / 1000000) *
                data[models[base_model]].append(frac)
                params[models[base_model]].append(score)
                print(f'Model {base_model} in epoch {epoch} with {imgs} images '
                      f'leads to score {score} with brain score {frac}')
        if percent > high_y:
            high_y = percent
    if len(selection) == 3:
        y = r"\textbf{mean(V4, IT, Behavior)}"
    else:
        y = r"\textbf{mean(V1,V2,V4,IT,Behavior)}"  # [\% of standard training]
    for i, ax in enumerate(axes):
        zero_indices = {key: np.array([tick == 0 for tick in xticks]) for key, xticks in params.items()}
        if i == 0:  # axis plotting the x=0 value
            ax_data = {key: np.array(values)[zero_indices[key]].tolist() for key, values in data.items()}
            xticks = {key: np.array(values)[zero_indices[key]].tolist() for key, values in params.items()}
            xticklabels = np.array([0])
            ylabel = y
        else:  # axis plotting everything x>0
            ax_data = {key: np.array(values)[~zero_indices[key]].tolist() for key, values in data.items()}
            xticks = {key: np.array(values)[~zero_indices[key]].tolist() for key, values in params.items()}
            xticklabels = np.array([.001, .01, .1, 1, 10, 100, 1000]) * 1000000
            ax.spines['left'].set_visible(False)
            ylabel = ''
        plot_data_double(ax_data, {}, '', x_name='',
                         x_labels=xticklabels, scatter=True, percent=percent,
                         alpha=0.8,
                         y_name=ylabel, x_ticks=xticks,
                         pal=['#2CB8B8', '#186363', '#ABB2B9', '#ABB2B9', '#ABB2B9', '#259C9C', '#36E3E3', '#9AC3C3'],
                         log=True,
                         x_ticks_2={}, ax=ax, million=True,
                         annotate_pos=0)

        # adopted from https://stackoverflow.com/a/32186074/2225200
        d = .015  # how big to make the diagonal lines in axes coordinates
        kwargs = dict(transform=ax.transAxes, color='#dedede', clip_on=False)
        if i == 0:
            m = 1 / .05
            ax.plot((1 - d * m, 1 + d * m), (-d, +d), **kwargs)
        else:
            kwargs.update(transform=ax.transAxes)
            ax.plot((-d, +d), (-d, +d), **kwargs)
            # remove yticks. We can't `ax.yaxis.set_visible(False)` altogether since that would also remove the grid
            for tic in ax.yaxis.get_major_ticks():
                tic.tick1On = tic.tick2On = False
            ax.set_yticklabels([])
        axes[0].set_ylim(axes[1].get_ylim())


def get_full(conn, convergence):
    return load_error_bared(conn, ['CORnet-S_full'], benchmarks, convergence)['CORnet-S_full']


if __name__ == '__main__':
    all_names = ['Artificial Genome+Critical Training (AG+CT)', ]
    mod = [{
        'CORnet-S_cluster2_v2_IT_trconv3_bi': 'IT.conv3_special',
        'CORnet-S_cluster2_v2_V4_trconv3_bi_seed42': 'V4.conv3_special',
        'CORnet-S_train_gmk1_cl2_7_7tr_bi_seed42': 'V2.conv3_special',
    }]
    selection = [2, 3, 4]
    plot_num_params_images(imagenet=False, entry_models=mod, all_labels=all_names, convergence=True,
                           images=[1000, 100000, 500000],
                           selection=selection)
    # plot_performance(False)
    # plot_num_params()
    # plot_num_params(True)
    # plot_performance(entry_model=best_special_brain)
    # plot_performance(False, entry_model=best_special_brain)
    # plot_figure_3()
    # plot_num_params(entry_model=best_special_brain)
    # plot_num_params(True, entry_model=best_special_brain)
