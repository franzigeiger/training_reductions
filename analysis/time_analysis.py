from itertools import chain

import numpy as np

from base_models.global_data import layer_best_2, random_scores, layers, convergence_epoch, benchmarks, \
    benchmarks_public, convergence_images
from benchmark.database import load_scores, get_connection, load_error_bared
from plot.plot_data import plot_data_base, plot_bar_benchmarks, scatter_plot, plot_data_double, red_palette, my_palette, \
    plot_heatmap, blue_palette, grey_palette


# layers = ['full', 'V1.conv1', 'V1.conv2',
#           'V2.conv_input', 'V2.skip', 'V2.conv1', 'V2.conv2', 'V2.conv3',
#           'V4.conv_input', 'V4.skip', 'V4.conv1', 'V4.conv2', 'V4.conv3',
#           'IT.conv_input', 'IT.skip', 'IT.conv1', 'IT.conv2', 'IT.conv3', 'decoder']


def plot_over_epoch(models):
    model_dict = {}
    conn = get_connection()
    epochs = (0, 5, 10, 15)
    for model in models:
        names = []
        for epoch in epochs:
            names.append(f'{model}_epoch_{epoch:02d}')
        model_dict[model] = load_scores(conn, names, benchmarks=benchmarks)
    model_dict[f'{model}_epoch_00'] = load_scores(conn, ['CORnet-S_random'], benchmarks)
    benchmarks_labels = ['V1', 'V2', 'V4', 'IT', 'Behavior', 'Imagenet']
    for i in range(6):
        data = {}
        for model in models:
            data[model] = []
            for epoch in epochs:
                data[model].append(model_dict[model][f'{model}_epoch_{epoch:02d}'][i])
        # data['CORnet-S'] = [0] * 3 + [model_dict['CORnet-S']['CORnet-S'][i]]
        plot_data_base(data, f'{benchmarks_labels[i]} Benchmark over epochs', epochs, 'Score over epochs', 'Score')


def plot_models_benchmarks(models, file_name, benchmarks, convergence=True, gs=None, ax=None):
    conn = get_connection()
    model_dict = load_error_bared(conn, models.keys(), benchmarks)
    if len(benchmarks) < 6:
        benchmarks_labels = ['V4', 'IT', 'Behavior', 'Imagenet']
    else:
        benchmarks_labels = ['V1', 'V2', 'V4', 'IT', 'Behavior']  # 'Imagenet'
    data_set = {}
    err = {}
    # We replace the model id, a more human readable version
    for id, desc in models.items():
        data_set[desc] = model_dict[id][:5]
        err[desc] = model_dict[id][6:-1]
    plot_bar_benchmarks(data_set, benchmarks_labels, '', r'\textbf{Scores}', file_name, yerr=err, gs=gs, ax=ax)


def plot_models_vs(models, file_name, convergence=False, epoch=0, imagenet=False, gs=None, ax=None, selection=[]):
    model_dict = {}
    conn = get_connection()

    names = []
    for name, mod in models.items():
        for model in mod.values():
            #         if convergence:
            #             epoch = convergence_epoch[model]
            names.append(model)
    model_dict = load_error_bared(conn, names, benchmarks, convergence=convergence, epochs=[0])
    labels = []
    data_set = {}
    err = {}
    # We replace the model id, a more human readable version
    for name, models in models.items():
        labels.append(name)
        # data_set[name] = {}
        for model_name, model in models.items():
            if model_name not in data_set:
                data_set[model_name] = []
                err[model_name] = []
            # if imagenet:
            #     data_set[model_name].append(np.mean(model_dict[f'{model}_epoch_{epoch:02d}'][5]))
            # else:
            if convergence:
                data_set[model_name].append(np.mean(model_dict[model][selection]))
                err[model_name].append(np.mean(model_dict[model][6:][selection]))
            else:
                data_set[model_name].append(np.mean(model_dict[f'{model}_epoch_{epoch:02d}'][selection]))
                err[model_name].append(np.mean(model_dict[f'{model}_epoch_{epoch:02d}'][6:][selection]))

    # if convergence and 'CORnet-S_full' in convergence_epoch:
    full_tr = load_error_bared(conn, [f'CORnet-S_full'], benchmarks)[
        f'CORnet-S_full']
    # else:
    #     full_tr = load_scores(conn, ['CORnet-S_full_epoch_0'], benchmarks)['CORnet-S_full_epoch_06']
    # print(f'Mean of brain benchmark model {desc}, {np.mean(data_set[desc][2:5])}')
    # if imagenet:
    #     line =np.mean(full_tr[5])
    # else:
    line = np.mean(full_tr[selection])
    if len(selection) == 3:
        y = r"\textbf{Brain Predictivity} "
    else:
        y = r"\textbf{Brain Predictivity}"
    pals = blue_palette
    plot_bar_benchmarks(data_set, labels, '', y, file_name, yerr=err, line=line, label=True, grey=False, gs=gs, ax=ax)


def plot_model_avg_benchmarks(models, file_name):
    model_dict = {}
    conn = get_connection()
    epoch = 6
    names = []
    for model in models.keys():
        names.append(f'{model}_epoch_{epoch:02d}')
    model_dict = load_scores(conn, names, benchmarks)
    benchmarks_labels = ['Brain Predictivity', 'Imagenet']
    data_set = {}
    # We replace the model id, a more human readable version
    for id, desc in models.items():
        data = model_dict[f'{id}_epoch_{epoch:02d}']
        data_set[desc] = [np.mean(data[0:5]), data[5]]
        print(f'Mean of brain benchmark model {desc}, {np.mean(data[0:5])}')
    plot_bar_benchmarks(data_set, benchmarks_labels, '', 'Scores', file_name)


def plot_models_benchmark_vs_public(models, file_name):
    model_dict = {}
    conn = get_connection()
    epoch = 6
    names = []
    for model in models.keys():
        names.append(f'{model}_epoch_{epoch:02d}')
    model_dict = load_scores(conn, names, benchmarks)
    model_dict_pub = load_scores(conn, names, benchmarks_public)
    benchmarks_labels = ['V1', 'V2', 'V4', 'IT', 'Behavior', 'Imagenet']
    data_set = {}
    # We replace the model id, a more human readable version
    for id, desc in models.items():
        data_set[desc] = model_dict[f'{id}_epoch_{epoch:02d}']
        data_set[f'{desc} public'] = model_dict_pub[f'{id}_epoch_{epoch:02d}']
        print(f'Mean of brain benchmark model {desc}, {np.mean(data_set[desc][2:5])}')
    plot_bar_benchmarks(data_set, benchmarks_labels, 'Model scores in epoch 6', 'Score [% of standard training]',
                        file_name, grey=True)


def plot_benchmarks_over_epochs(model, epochs=None, benchmarks=benchmarks, selection=[2, 3, 4], ax=None):
    # TODO: Add error bars
    model_dict = {}
    conn = get_connection()
    if epochs is None:
        epochs = (0, 5, 10, 15, 20)

    names = []
    # for epoch in epochs:
    #     if epoch % 1 == 0:
    #         names.append(f'{model}_epoch_{epoch:02d}')
    #     else:
    #         names.append(f'{model}_epoch_{epoch:.1f}')
    model_dict = load_error_bared(conn, [model, 'CORnet-S_full'], benchmarks, epochs=epochs, convergence=True)
    full = model_dict['CORnet-S_full']

    benchmarks_labels = ['V1', 'V2', 'V4', 'IT', 'Behavior', 'Mean']
    data = {}
    for i in range(len(benchmarks) - 1):
        if i in selection:
            data[benchmarks_labels[i]] = []
            for epoch in epochs:
                if epoch % 1 == 0:
                    frac = (model_dict[f'{model}_epoch_{epoch:02d}'][i] / full[i]) * 100
                else:
                    frac = (model_dict[f'{model}_epoch_{epoch:.1f}'][i] / full[i]) * 100
                data[benchmarks_labels[i]].append(frac)
        end = (np.mean(model_dict[model][i]) / np.mean(full[i])) * 100
        data[benchmarks_labels[i]].append(end)
    data[benchmarks_labels[-1]] = []
    for epoch in epochs:
        if epoch % 1 == 0:
            frac = (np.mean(model_dict[f'{model}_epoch_{epoch:02d}'][selection]) / np.mean(full[selection])) * 100
        else:
            frac = (np.mean(model_dict[f'{model}_epoch_{epoch:.1f}'][selection]) / np.mean(full[selection])) * 100
        data[benchmarks_labels[-1]].append(frac)
    end = (np.mean(model_dict[model][selection]) / np.mean(full[selection])) * 100
    data[benchmarks_labels[-1]].append(end)
    plot_data_base(data, f'', r'\textbf{Epoch}', r'\textbf{Score} [\% of standard training]',
                   epochs + [43],
                   x_ticks=[value for value in epochs if value not in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 15]] + [
                       43],
                   x_labels=[value for value in epochs if value not in [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 15]] + [
                       'Conv'],
                   percent=True, alpha=0.5, log=True, annotate=True, legend=False, annotate_pos=2, ax=ax,
                   palette=grey_palette[:len(benchmarks_labels) - 1] + [blue_palette[0]])


def plot_first_epochs(models, epochs=None, brain=True, convergence=True, ax=None):
    model_dict = {}
    conn = get_connection()
    if epochs is None:
        epochs = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6)
    data = {}
    x_values = {}
    if convergence and 'CORnet-S_full' in convergence_epoch:
        full_tr = load_scores(conn, [f'CORnet-S_full_epoch_{convergence_epoch["CORnet-S_full"]:02d}'], benchmarks)[
            f'CORnet-S_full_epoch_{convergence_epoch["CORnet-S_full"]:02d}']
    else:
        full_tr = load_scores(conn, ['CORnet-S_full_epoch_06'], benchmarks)['CORnet-S_full_epoch_06']
    for model, name in models.items():
        names = []
        for epoch in epochs:
            if epoch % 1 == 0:
                names.append(f'{model}_epoch_{epoch:02d}')
            else:
                names.append(f'{model}_epoch_{epoch:.1f}')
        if convergence and model in convergence_epoch:
            names.append(f'{model}_epoch_{convergence_epoch[model]:02d}')
        model_dict = load_scores(conn, names, benchmarks)
        scores = []
        for epoch in epochs:
            if brain:
                full = np.mean(full_tr[2:5])
                if epoch % 1 == 0:
                    frac = (np.mean(model_dict[f'{model}_epoch_{int(epoch):02d}'][2:5]) / full) * 100
                    scores.append(frac)
                else:
                    frac = (np.mean(model_dict[f'{model}_epoch_{epoch:.1f}'][2:5]) / full) * 100
                    scores.append(frac)
            else:
                full = np.mean(full_tr[5])
                if epoch % 1 == 0:
                    frac = (np.mean(model_dict[f'{model}_epoch_{int(epoch):02d}'][5]) / full) * 100
                    scores.append(frac)
                else:
                    frac = (np.mean(model_dict[f'{model}_epoch_{epoch:.1f}'][5]) / full) * 100
                    scores.append(frac)
        if convergence and model in convergence_epoch:
            if brain:
                frac = (np.mean(model_dict[f'{model}_epoch_{convergence_epoch[model]:02d}'][2:5]) / full) * 100
                scores.append(frac)
            else:
                frac = (np.mean(model_dict[f'{model}_epoch_{convergence_epoch[model]:02d}'][5]) / full) * 100
                scores.append(frac)
            x_values[name] = epochs + [convergence_epoch[model]]
        else:
            x_values[name] = epochs
        data[name] = scores

    title = f'Brain scores mean vs epochs' if brain else 'Imagenet score vs epochs'
    plot_data_base(data, 'First epochs', x_values, 'Epochs', 'Brain Predictivity [% of standard training]',
                   x_ticks=epochs + [30, 40, 50], log=True,
                   percent=True, special_xaxis=True, legend=False, only_blue=False, palette=red_palette, annotate=True,
                   annotate_pos=1, ax=ax)


def plot_single_benchmarks(models, epochs=None, compare_batchfix=False, run_mean=False):
    model_dict = {}
    conn = get_connection()
    if epochs is None:
        epochs = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6)
    data = {}
    benchmarks_label = ['V1', 'V2', 'V4', 'IT', 'Behavior', 'Imagenet']

    for model, name in models.items():
        names = []
        for epoch in epochs:
            if epoch % 1 == 0:
                names.append(f'{model}_epoch_{epoch:02d}')
            else:
                names.append(f'{model}_epoch_{epoch:.1f}')

            if compare_batchfix:
                names.append(f'{model}_epoch_{epoch:02d}_BF')
        model_dict[name] = load_scores(conn, names, benchmarks)
    for i in range(6):
        for model, name in models.items():
            scores = []
            for epoch in epochs:
                if epoch % 1 == 0:
                    scores.append(model_dict[name][f'{model}_epoch_{int(epoch):02d}'][i])
                else:
                    scores.append(model_dict[name][f'{model}_epoch_{epoch:.1f}'][i])

            if run_mean:
                data[name] = [scores[0]] + np.convolve(scores, np.ones((3,)) / 3, mode='valid') + [scores[-1]]
            else:
                data[name] = scores

            if compare_batchfix:
                scores = []
                for epoch in epochs:
                    scores.append(model_dict[name][f'{model}_BF_epoch_{epoch:02d}'][i])
                    if run_mean:
                        data[f'{name}_BF'] = np.convolve(scores, np.ones((3,)) / 3, mode='same')
                    else:
                        data[f'{name}_BF'] = scores

        title = f'{benchmarks_label[i]} benchmark vs epochs'
        plot_data_base(data, title, epochs, 'Epoch', 'Score', x_ticks=epochs, log=True)


def score_over_layers(models, random, labels, bench, convergence=True, ax=None):
    if bench is not None:
        benchmarks = bench
    conn = get_connection()
    names = []
    if convergence and 'CORnet-S_full' in convergence_epoch:
        full_tr = load_scores(conn, [f'CORnet-S_full_epoch_{convergence_epoch["CORnet-S_full"]:02d}'], benchmarks)[
            f'CORnet-S_full_epoch_{convergence_epoch["CORnet-S_full"]:02d}']
    else:
        full_tr = load_scores(conn, ['CORnet-S_full_epoch_06'], benchmarks)['CORnet-S_full_epoch_06']
    model_dict = load_error_bared(conn, list(chain(models.keys(), random.keys())), benchmarks, convergence=convergence)
    if len(benchmarks) < 6:
        benchmarks_labels = ['V4', 'IT', 'Behavior', 'Imagenet']
    else:
        benchmarks_labels = ['V1', 'V2', 'V4', 'IT', 'Behavior', 'Imagenet']
    data = {}
    err = {}
    x_ticks = {}
    for i in range(len(benchmarks)):
        data[benchmarks_labels[i]] = []
        err[benchmarks_labels[i]] = []
        x_ticks[benchmarks_labels[i]] = []
        layers = []
        for model, layer in models.items():
            layers.append(layer_best_2[layer])
            frac = (model_dict[model][i] / full_tr[i]) * 100
            frac_err = (model_dict[model][len(benchmarks):][i] / full_tr[i]) * 100
            data[benchmarks_labels[i]].append(frac)
            err[benchmarks_labels[i]].append(frac_err)
        x_ticks[benchmarks_labels[i]] = layers
    plot_data_double(data, data2=None, err=err, name=f'Artificial Genome + Critical Training',
                     x_name='Number of trained layers',
                     y_name=r'Benchmark Score [% of standard training]',
                     x_ticks=x_ticks, x_ticks_2=[], percent=True, ax=ax, pal=red_palette, annotate_pos=1)


def score_over_layers_avg(all_models, random, all_labels=[], imagenet=False, convergence=False, ax=None, selection=[]):
    conn = get_connection()
    data = {}
    err2 = {}
    full = 0
    names = []
    for models in all_models:
        names.extend(models.keys())
    names.extend(random.keys())
    model_dict = load_error_bared(conn, names, benchmarks, convergence=convergence)
    data2 = {}
    data2['Score'] = []
    err2['Score'] = []
    layers2 = []
    for model, layer in random.items():
        layers2.append(layer_best_2[layer])
        if model == 'CORnet-S_full':
            full = np.mean(model_dict[model][selection])
            full_err = (np.mean(model_dict[model][6:][selection]) / full) * 100
            data2['Score'].append(100)
            err2['Score'].append(full_err)
        else:
            percent = (np.mean(model_dict[model][selection]) / full) * 100
            percent_error = (np.mean(model_dict[model][6:][selection]) / full) * 100
            data2['Score'].append(percent)
            err2['Score'].append(percent_error)
    x_ticks = {}
    labels = {}
    err = {}
    for models, name in zip(all_models, all_labels):
        data[name] = []
        err[name] = []
        layers = []
        for model, layer in models.items():
            if convergence and model in convergence_epoch:
                postfix = f'_epoch_{convergence_epoch[model]:02d}'
            else:
                postfix = f'_epoch_06'
            layers.append(layer_best_2[layer])
            if model == 'CORnet-S_full':
                full = np.mean(model_dict[model][selection])
                data[name].append(100)
            else:
                percent = (np.mean(model_dict[model][selection]) / full) * 100
                data[name].append(percent)
            full_err = (np.mean(model_dict[model][6:][selection]) / full) * 100
            err[name].append(full_err)
        x_ticks[name] = layers
        # short = name.split('(')[1][:-1]
        # labels[name] = [f'{value.split(".")[0]}_{short}' for value in models.values()]
        labels[name] = [name]

    if imagenet:
        title = f'Imagenet over layers'
        y = r"Imagenet}[% of standard training]"
    else:
        title = f'Brain-Score Benchmark mean(V4, IT, Behavior) over layers'
        if len(selection) == 3:
            y = r"Brain Predictivity [% of standard training]"
        else:
            y = r"Brain Predictivity [% of standard training]"

    plot_data_double(data, data2, '', err=err, err2=err2, x_name='Number of trained layers',
                     y_name=y, x_ticks=x_ticks,
                     x_ticks_2=layers2, percent=True, annotate_pos=0, pal=my_palette, ax=ax)


def image_scores(models, imgs, labels, ax=None, selection=[]):
    names = []
    conn = get_connection()
    for model in models:
        for img in imgs:
            name = f'{model}_img{img}'
            names.append(f'{name}_epoch_{convergence_images[name]}')
        if model == 'CORnet-S_cluster2_v2_IT_trconv3_bi':
            model = f'{model}_seed42'
        names.append(f'{model}_epoch_{convergence_epoch[model]}')
    names.append('CORnet-S_full_epoch_43')
    model_dict = load_scores(conn, names, benchmarks)
    data2 = {}
    full = np.mean(model_dict['CORnet-S_full_epoch_43'][selection])
    for i in imgs:
        for model, name in zip(models, labels):
            data2[name] = []
            name1 = f'{model}_img{i}'
            frac = (np.mean(model_dict[f'{name1}_epoch_{convergence_images[name1]}'][selection]) / full) * 100
            data2[name].append(frac)
            if model == 'CORnet-S_cluster2_v2_IT_trconv3_bi':
                model = f'{model}_seed42'
            frac = (np.mean(model_dict[f'{model}_epoch_{convergence_epoch[model]}'][selection]) / full) * 100
            data2[name].append(frac)

    if len(selection) == 1:
        title = 'Imagenet score vs number of weights'
        y = r'Imagenet [% of standard training]'
    else:
        title = f'Brain scores mean vs number of weights'
        y = r'Brain Predictivity [% of standard training]'
    imgs.append(1200000)
    plot_data_double(data2, {}, '', x_name='Number of images in million', y_name=y,
                     x_ticks={'IT init, selective training': imgs},
                     x_ticks_2=imgs, percent=True, log=True, ax=ax, million=True)


def image_scores_single(model, imgs, selection=[], ax=None):
    names = []
    conn = get_connection()
    for img in imgs:
        name = f'{model}_img{img}'
        names.append(name)
    # if model == 'CORnet-S_cluster2_v2_IT_trconv3_bi':
    #     model = f'{model}_seed42'
    # names.append(f'{model}_epoch_{convergence_epoch[model]}')
    names.append('CORnet-S_full')
    # model_dict = load_scores(conn, names, benchmarks)
    model_dict = load_error_bared(conn, names, benchmarks, convergence=True)
    full = model_dict[model]
    benchmarks_labels = ['V1', 'V2', 'V4', 'IT', 'Behavior', 'Mean']
    data = {}
    for i in range(len(benchmarks) - 1):
        if i in selection:
            data[benchmarks_labels[i]] = []
            for j in imgs:
                name1 = f'{model}_img{j}'
                frac = (np.mean(model_dict[name1][i]) / full[i]) * 100
                data[benchmarks_labels[i]].append(frac)
            frac = (np.mean(model_dict[model][i]) / full[i]) * 100
            data[benchmarks_labels[i]].append(frac)
    data[benchmarks_labels[-1]] = []
    for j in imgs:
        name1 = f'{model}_img{j}'
        frac = (np.mean(model_dict[name1][selection]) / np.mean(full[selection])) * 100
        data[benchmarks_labels[-1]].append(frac)
    frac = (np.mean(model_dict[model][selection]) / np.mean(full[selection])) * 100
    data[benchmarks_labels[-1]].append(frac)
    imgs.append(1280000)
    plot_data_base(data, '', r'\textbf{Images} [Million]', r'\textbf{Score} [\% of standard training]', x_values=imgs,
                   x_ticks=[100, 1000, 10000, 100000, 1280000], x_labels=['100', '1k', '10k', '100k', '1.3M'],
                   million=True, palette=grey_palette[:len(benchmarks_labels) - 1] + [blue_palette[0]], alpha=0.5,
                   use_xticks=True,
                   percent=True, log=True, annotate=True, legend=False, annotate_pos=3, ax=ax)


def image_epoch_heatmap(model, imgs, epochs, selection=[], title=r'\textbf{Standard training epochs/images trade-off}',
                        ax=None):
    names = []
    conn = get_connection()
    # delta= 'CORnet-S_full'
    # if model == 'CORnet-S_cluster2_v2_IT_trconv3_bi':
    #     model1 = model
    # else:
    #     model1 = model
    for img in imgs:
        name = f'{model}_img{img}'
        # for epoch in epochs:
        #     names.append(f'{name}_epoch_{epoch:02d}')
        names.append(name)
    names.append(model)
    # for epoch in epochs:
    #     names.append(f'{model}_epoch_{epoch:02d}')
    names.append('CORnet-S_full')
    model_dict = load_error_bared(conn, names, epochs=epochs, benchmarks=benchmarks)
    full = np.mean(model_dict['CORnet-S_full'][selection])
    matrix = np.zeros([len(imgs) + 1, len(epochs) + 1])
    data = {}
    for i in range(len(imgs)):
        for j in range(len(epochs)):
            name1 = f'{model}_img{imgs[i]}_epoch_{epochs[j]:02d}'
            frac = (np.mean(model_dict[name1][selection]) / full)
            matrix[i, j] = frac
        name = f'{model}_img{imgs[i]}'
        # name = f'{name}_epoch_{convergence_images[name]:02d}'
        frac = (np.mean(model_dict[name][selection]) / full)
        matrix[i, -1] = frac

    # names.append(f'{model1}_epoch_{convergence_epoch[model1]:02d}')
    for j in range(len(epochs)):
        name1 = f'{model}_epoch_{epochs[j]:02d}'
        frac = (np.mean(model_dict[name1][selection]) / full)
        matrix[-1, j] = frac
    # name = f'{model}_epoch_{convergence_epoch[model]:02d}'
    frac = (np.mean(model_dict[model][selection]) / full)
    matrix[-1, -1] = frac
    plot_heatmap(matrix, r'\textbf{Epochs}', r'\textbf{Images}', title=title, annot=True, ax=ax,
                 cbar=False, cmap='YlOrRd', percent=False,
                 fmt='.0%', vmin=0, vmax=1, yticklabels=imgs + ['All'], xticklabels=epochs + ['Convergence'], alpha=0.8)
    # plot_data_base(data, f'CORnet-S Brain-Score benchmarks', imgs, 'Epoch', r'Score [% of standard training]',
    #                x_ticks=imgs, percent=True,log=True, annotate=True, legend=False, annotate_pos=3, ax=ax)


def delta_heatmap(model1, model2, imgs, epochs, selection=[], title='', ax=None):
    names = []
    conn = get_connection()
    for model in [model1, model2]:
        if model == 'CORnet-S_cluster2_v2_IT_trconv3_bi':
            model_spec = model
        else:
            model_spec = model
        for img in imgs:
            name = f'{model}_img{img}'
            for epoch in epochs:
                names.append(f'{name}_epoch_{epoch:02d}')
            names.append(f'{name}_epoch_{convergence_images[name]}')
        names.append(f'{model_spec}_epoch_{convergence_epoch[model_spec]}')
        for epoch in epochs:
            names.append(f'{model}_epoch_{epoch:02d}')
        names.append('CORnet-S_full_epoch_43')
    model_dict = load_scores(conn, names, benchmarks)
    full = np.mean(model_dict['CORnet-S_full_epoch_43'][selection])
    matrix = np.zeros([len(imgs) + 1, len(epochs) + 1])
    data = {}
    for i in range(len(imgs)):
        for j in range(len(epochs)):
            name1 = f'{model1}_img{imgs[i]}_epoch_{epochs[j]:02d}'
            name2 = f'{model2}_img{imgs[i]}_epoch_{epochs[j]:02d}'
            matrix[i, j] = calc_dif(name1, name2, model_dict, full, selection)
        name = f'{model1}_img{imgs[i]}'
        name = f'{name}_epoch_{convergence_images[name]:02d}'
        name2 = f'{model2}_img{imgs[i]}'
        name2 = f'{name2}_epoch_{convergence_images[name2]:02d}'
        matrix[i, -1] = calc_dif(name, name2, model_dict, full, selection)
    names.append(f'{model1}_epoch_{convergence_epoch[model1]:02d}')
    for j in range(len(epochs)):
        name1 = f'{model1}_epoch_{epochs[j]:02d}'
        name2 = f'{model2}_epoch_{epochs[j]:02d}'
        matrix[-1, j] = calc_dif(name1, name2, model_dict, full, selection)
    name = f'CORnet-S_cluster2_v2_IT_trconv3_bi_epoch_{convergence_epoch["CORnet-S_cluster2_v2_IT_trconv3_bi"]:02d}'
    name2 = f'{model2}_epoch_{convergence_epoch[model2]:02d}'
    matrix[-1, -1] = calc_dif(name, name2, model_dict, full, selection)
    plot_heatmap(matrix, r'\textbf{Epochs}', r'\textbf{Images}',
                 title=title, annot=True, ax=ax,
                 cbar=False, cmap='RdYlGn', percent=False, alpha=0.6,
                 fmt='.0%', vmin=-0.30, vmax=0.30, yticklabels=imgs + ['All'], xticklabels=epochs + ['Convergence'])


def calc_dif(name1, name2, model_dict, full, selection):
    frac1 = (np.mean(model_dict[name1][selection]) / full)
    frac2 = (np.mean(model_dict[name2][selection]) / full)
    return frac1 - frac2


def score_layer_depth(values, brain=True):
    names = []
    conn = get_connection()
    for k, v in values.items():
        names.append(f'{k}_epoch_05')
    for k, v in random_scores.items():
        if k != 'CORnet-S_random' and k != 'CORnet-S_train_random':
            names.append(f'{k}_epoch_05')
        else:
            names.append(k)
    model_dict = load_scores(conn, names,
                             ['movshon.FreemanZiemba2013.V1-pls',
                              'movshon.FreemanZiemba2013.V2-pls',
                              'dicarlo.Majaj2015.V4-pls',
                              'dicarlo.Majaj2015.IT-pls',
                              'dicarlo.Rajalingham2018-i2n',
                              'fei-fei.Deng2009-top1'])
    weight_num = [9408, 36864, 8192, 16384, 65536, 2359296, 65536, 32768, 65536, 262144, 9437184, 262144, 131072,
                  262144, 1048576, 37748736, 1048576, 512000]
    acc = [52860096 + 512000]
    for i in weight_num:
        acc.append(acc[-1] - i)
    weights = []
    results = []
    for model, l in values.items():
        index = layers.index(l)
        weights.append(acc[index])
        res = model_dict[f'{model}_epoch_05']
        if brain:
            results.append(np.mean(res[2:4]))
            # if index < 7:
            #     results.append(np.mean(res[0:1]))
            # else:
            #     results.append(np.mean(res[0:2]))
        else:
            results.append(res[5])
    rand_names = []
    for model, l in random_scores.items():
        index = layers.index(l)
        weights.append(acc[index])
        if model != 'CORnet-S_random' and model != 'CORnet-S_train_random':
            res = model_dict[f'{model}_epoch_05']
        else:
            res = model_dict[model]

        if brain:
            results.append(np.mean(res[0:4]))
            # if index < 7:
            #     results.append(np.mean(res[0:1]))
            # else:
            #     results.append(np.mean(res[0:2]))
        else:
            results.append(res[5])
        rand_names.append(f'Random {l}')
    title = f'Brain scores mean vs number of weights' if brain else 'Imagenet score vs number of weights'
    scatter_plot(weights, results, x_label='Num of weights', y_label='Score', labels=list(values.values()) + rand_names,
                 title=title)


models = {
    # Batchnrom corrected
    # Layer 1
    # 'CORnet-S_train_V2': 'V1 random train after',
    # 'CORnet-S_train_gabor_multi_dist' : 'Gabor multidimensional distribution ',
    # 'CORnet-S_train_gabor_scrumble' : 'Gabor scrumble existing parameter',
    # 'CORnet-S_train_gabor_dist' : 'Gabor per parameter distribution',

    # #  Layer 1 & 2

    # 'CORnet-S_train_gabor_dist_weight_dist_channel' : 'V1 base',
    # 'CORnet-S_train_gabor_dist_weight_dist_channel_ra_BF' : 'V1 no batchnorm',
    # 'CORnet-S_train_gabor_dist_weight_dist_channel_bi_BF' : 'V1 batchnorm',
    # 'CORnet-S_train_gabor_dist_both_kernel' : "V1 gabor ",
    # 'CORnet-S_train_gabor_dist_both_kernel_ra_BF' : "V1 gabor no batchnorm",
    # 'CORnet-S_train_gabor_dist_both_kernel_bi_BF' : "V1 gabor batchnorm", # best

    # 'CORnet-S_train_gabor_dist_both_kernel_bi_BF': 'V1 gabor',
    # 'CORnet-S_train_wmk0_wmc1_bi': 'V1 kernel dist',
    # 'CORnet-S_train_wmc0_wmc1_bi': 'V1 channel dist',
    # 'CORnet-S_train_kn1_kn2_bi_v2': 'V1 kernel normal',
    # 'CORnet-S_train_ln1_ln2_bi': 'V1 layer normal',
    # 'CORnet-S_train_cn1_cn2_bi': 'V1 channel normal',
    # 'CORnet-S_train_gmk1_bd2_bi': 'Gabor plus best dist',
    # 'CORnet-S_train_gmk1_cl2_bi': 'Cluster l2',

    # Layer 3
    # 'CORnet-S_train_gmk1_wmc2_ln3' : "V2.conv1 Layer norm dist",
    # 'CORnet-S_train_gmk1_wmc2_ln3_bi_BF' : "V2.conv1 Layer bn",
    # 'CORnet-S_train_gmk1_wmc2_ln3_ra_BF' : "V2.conv1 Layer no bn",
    # 'CORnet-S_train_gmk1_wmc2_kn3' : "V2.conv1 Kernel norm dist",
    # 'CORnet-S_train_gmk1_wmc2_kn3_bi_BF' : "V2.conv1 Kernel bn",
    # 'CORnet-S_train_gmk1_wmc2_kn3_ra_BF' : "V2.conv1 Kernel no bn",
    #
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    # 'CORnet-S_train_gmk1_gmk2_ln3_bi': 'V2.input layer dist batchnorm',  # best
    # 'CORnet-S_train_gmk1_gmk2_ln3_ra' : 'V2.input layer dist no batchnorm',
    # 'CORnet-S_train_gmk1_gmk2_kn3_bi' : 'V2.input kernel norm batchnorm',
    # 'CORnet-S_train_gmk1_gmk2_kn3_ra' : 'V2.input kernel norm no batchnorm',

    # Layer 4
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_bi' : 'V2.skip Kernel norm dist bn',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_ra' : 'V2.skip Kernel norm dist no bn',
    # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_bi': 'V2.skip Layer norm dist bn',  # best
    # # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_ra' : 'V2.skip Layer norm dist no bn',
    # 'CORnet-S_train_gmk1_gmk2_ln3_ln4_bi' : 'V2.skip Layer norm dist bn 2',
    # 'CORnet-S_train_gmk1_gmk2_ln3_kn4_bi' : 'V2.skip Layer norm dist bn 3',

    # Layer 5
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    # 'CORnet-S_train_gmk1_gmk2_ln3_ln4_ln5_ra': 'V2.conv1 layer norm dist',
    # 'CORnet-S_train_gmk1_gmk2_ln3_ln4_ln5_bi': 'V2.conv1 layer norm dist batchnorm',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_ra': 'V2.conv1 kernel norm dist',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_bi': 'V2.conv1 kernel norm dist batchnorm',
    #  'CORnet-S_train_gmk1_gmk2_ln3_kn4_ln5_bi': 'V2.conv1', # best
    #  'CORnet-S_train_gmk1_gmk2_kn3_ln4_kn5_bi': 'V2.conv1 layer norm dist batchnorm 3',
    #  'CORnet-S_train_gmk1_gmk2_kn3_ln4_ln5_bi': 'V2.conv1 layer norm dist batchnorm 4',

    # # # # Layer 6
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_bi': 'Best layer 5',
    # # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_ln5_wm6_ra': 'V2.conv2 weight dist',
    # # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_ln5_wm6_bi': 'V2.conv2 weight dist batchnorm',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_bi': 'Layer 6 weight distribution',
    # # 'CORnet-S_train_gmk1_gmk2_ln3_ln4_ln5_wm6_ra': 'V2.conv2 weight dist V2',
    # # 'CORnet-S_train_gmk1_gmk2_ln3_ln4_ln5_wm6_bi' : 'V2.conv2 weight dist V3 batchnorm',
    # # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_kn5_wmc6_bi' : 'V2.conv2 weight dist V4 batchnorm',
    # # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_ln5_wmc6_bi' : 'V2.conv2',

    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_kn6_kn7train_bi': 'Layer 7 kernel normal + train',

    # 'CORnet-S_train_gabor_dist_both_kernel_bi_BF': 'V1',
    # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_ln5_wmc6_bi' : 'V2.conv2 gabor',
    # # 'CORnet-S_train_wmk1_wmk2_kn3_ln4_ln5_wmc6_bi':'V2.conv2 kernel dist',
    # # 'CORnet-S_train_wmc1_wmc2_kn3_ln4_ln5_wmc6_bi': 'V2.conv2 channel dist',
    # 'CORnet-S_train_kn1_kn2_kn3_ln4_ln5_wmc6_bi':'V2.conv2 new mixture channel',
    # 'CORnet-S_train_kn1_kn2_kn3_ln4_ln5_wmk6_bi':'V2.conv2 new mixture kernel',
    # 'CORnet-S_train_kn1_kn2_kn3_kn4_kn5_kn6_bi_v2' : 'V2.conv2 kernel normal',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_prev6' : 'Previous std',

    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_beta6' : 'Beta dist',
    # # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_uniform6' : 'Uniform dist',
    # # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_poisson6' : 'Poisson dist',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_kn6_bi': 'Layer 6 kernel normal',
    # # 'CORnet-S_train_gmk1_gmk2_kn3_mi4_kn5_kn6' : 'Mutual information skip layer',
    # # 'CORnet-S_train_gmk1_gmk2_uniform3-7' : 'Uniform dist layer 3-7',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_ev6' : 'Eigenvalue layer 6',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_evd6' : 'Eigenvalue dist layer 6',
    # 'CORnet-S_train_evd1_evd2_kn3_kn4_kn5_evd6_bi' : 'Eigenvalue dist first layers',
    # 'CORnet-S_train_ev1_ev2_kn3_kn4_kn5_ev6_bi' : 'Eigenvalue first layers',
    # 'CORnet-S_train_gmk1_gmk2_wmk3_5_wmc6' : 'All kernel gaussian',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_best6' : 'Layer 6 kn',
    # 'CORnet-S_train_gmk1_gmk2_kn3_mi4_kn5_bd6_bi': 'Best dist + mi',
    # 'CORnet-S_train_gmk1_gmk2_kn3_mi4_kn5_bd6_bi' : 'best dist other 6',#*best

    # 'CORnet-S_train_gmk1_gmk2_kn3_mi4_kn5_bd6_bi' : 'Layer 6 ',#*best
    # 'CORnet-S_train_gmk1_gmk2_kn3_mi4_kn5_bd6_kn7_bi' : 'Add layer 7 kn', #*best 0.4676362894628521
    # 'CORnet-S_train_gmk1_gmk2_bd3_6_bi' : 'Layer 6 best dist' ,
    # 'CORnet-S_train_gmk1_gmk2_bd3_7_bi' : 'Add layer 7 best dist',

    # layer 7
    # 'CORnet-S_train_kn1_kn2_kn3_kn4_kn5_kn6_kn7_bi' : 'V2.conv3 new',
    # 'CORnet-S_train_V4': 'Random init V2',

    # 'CORnet-S_train_gmk1_gmk2_ln3_kn4_ln5_wmk6_kn7_bi' : 'Layer 7 bn',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wmk6_kn7_bi' : 'Layer 7 wk bn',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_kn6_kn7_bi' : 'Layer 7 kernel normal',

    # 'CORnet-S_brain_kn8_kn9_kn10_wmk11_kn12_bi' : 'Layer 12  V1 bn',
    # 'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_bi' : 'Layer 12 V2  bn',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_kn6_kn7_bi' : 'Layer kn7 2',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_ev6_kn7' : 'Eigenvalue layer 7',
    # 'CORnet-S_train_gmk1_gmk2_kn3_mi4_kn5_bd6_kn7_bi' : 'Layer 7 best dist',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wmc6_kn7_bi' : 'Layer 7 kn',
    # # 'CORnet-S_train_gmk1_gmk2_bd3_mi4_bd5_evd6_bd7_bi' : 'best dist, mutual information, eigenvalues',
    # 'CORnet-S_train_gmk1_gmk2_bd3_7_bi' : 'best dist layer 7',
    # 'CORnet-S_train_gmk1_gmk2_bd3_mi4_bd57_bi' : 'best dist mutual inf',
    # 'CORnet-S_train_gmk1_gmk2_kn3_mi4_kn5_bd6_kn7_bi' : 'best dist other layer 7', #*best 0.4676362894628521
    # 'CORnet-S_train_gmk1_gmk2_kn3_mi4_kn5_wmc6_kn7_bi' : 'Layer 7 brain mutual',
    # 'CORnet-S_train_gmk1_gmk2_bd3_bd4_bd5_wmc6_bd7_bi' : 'Layer 7 brain best dist',
    # 'CORnet-S_train_gmk1_gmk2_bd3_mi4_bd5_wmc6_bd7_bi' : 'Layer 7 brain mi bd',
    # 'CORnet-S_train_gmk1_cl2_bd3_mi4_bd5_wcl6_bd7_bi' : 'LAyer 7 cluster',

    # # Layer 8 upwards
    # 'CORnet-S_train_IT_seed_0' : 'Random init V4',
    # # 'CORnet-S_brain2_kn8_kn9_kn10_kn11_bi' : 'V4.conv2',
    # # 'CORnet-S_brain2_kn8_kn9_kn10_kn11_kn12_bi' : 'V4.conv3',
    # # 'CORnet-S_brain2_knall_IT_bi' : 'IT.conv3',
    # # 'CORnet-S_brain2_t7_kn8_kn9_kn10_kn11_bi' : 'V4.conv2',
    # # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_ln5_wmc6_bi' : 'V2.conv2 gabor',
    # #
    # 'CORnet-S_brain3_kn8_kn9_kn10_kn11_kn12_tra_bi' : 'V4.conv3 train 1',
    # 'CORnet-S_brain_kn8_kn9_kn10_wmk11_kn12_tr_bi' : 'V4.conv3 train2',
    # 'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_tr_bi': 'V4.conv3 train3',
    # 'CORnet-S_brain4_kn8_kn9_kn10_kn11_kn12_tra_bi' : 'V4.conv3 train4',
    # 'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_tr_bi'
    # 'CORnet-S_brain_kn8_kn9_kn10_kn11_kn12_tr_bi' : 'V4.conv3 train4',
    #     'CORnet-S_brain4_kn8_kn9_kn10_kn11_kn12_tra_bi' : 'V4.conv3 train 2',
    # 'CORnet-S_brain3_kn8_kn9_kn10_kn11_kn12_bi' : 'V4.conv3 ',
    # 'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_tr_bi' : 'V4.conv3 train 3',
    # 'CORnet-S_best_bd_mi_V4_trconv3_bi': 'V4.conv3',
    # 'CORnet-S_mix2_V4_trconv3_bi' : 'V4.conv3 mix2',
    # 'CORnet-S_dist_allbd_V4_trconv3_bi' : 'V4.conv3 dist',
    # 'CORnet-S_mix_bd_evd_mi_V4_trconv3_bi' : 'V4.conv3 mix',
    # 'CORnet-S_brainbest_V4_trconv3_bi': 'V4.conv3 brainbest',
    # 'CORnet-S_brainmutual_V4_trconv3_bi': 'V4.conv3 brainmutual',
    # 'CORnet-S_brainboth_V4_trconv3_bi' : 'V4.conv3 brainboth', #*best 0.4547913626969223
    #
    'CORnet-S_train_random': 'Random init IT',
    # # 'CORnet-S_brain2_t7_t12_knall_IT_bi_v2' : 'IT.conv2 new',
    # 'CORnet-S_brain3_t7_t12_knall_IT_bi': 'IT.conv2, conv3 train',
    # 'CORnet-S_brain_t7_t12_knk15_IT_bi': 'IT.conv2, conv3 train1',
    # # 'CORnet-S_brain3_knall_IT_bi': 'IT.conv3 no train',
    # 'CORnet-S_brain3_knall_IT.conv2_bi': 'IT.conv2 no train',
    # 'CORnet-S_brain_t7_t12_wmc15_IT_bi' : 'Weight mixture channel',
    # 'CORnet-S_brain3_knall_IT_bi_full': 'IT.conv3 full train',
    # 'CORnet-S_brain_t7_t12_wmc15_IT_bi': 'IT.conv3 MG',  # 0.4120501992349898
    # # 'CORnet-S_mixture_all_trconv3_bi': 'IT.conv3 mixture',
    # # 'CORnet-S_mix_bd_evd_mi_IT_trconv3_bi': 'IT.conv3 mix',
    # 'CORnet-S_best_bd_mi_IT_trconv3_bi' : 'IT best conv3',
    # # 'CORnet-S_dist_allbd_IT_trconv3_bi': 'IT conv3 distr',
    # 'CORnet-S_mix2_trconv3_bi': 'IT conv3 mix2',
    # 'CORnet-S_brainbest_IT_trconv3_bi': 'IT.conv3 Best distribution',  # 0.4079878622341095
    # 'CORnet-S_brainboth_IT_trconv3_bi': 'IT.conv3 brainboth',  # 0.4067269833226943
    # 'CORnet-S_brainmutual_IT_trconv3_bi': 'IT.conv3 brainmutual',
    # 'CORnet-S_brainbest2_IT_trconv3_bi': 'IT.conv3 brainbest2',
    # 'CORnet-S_brainboth2_IT_trconv3_bi': 'IT.conv3 brainboth2',
    # 'CORnet-S_cluster2_v2_IT_trconv3_bi': 'IT.conv3 Cluster',
    # 'CORnet-S_cluster5_v2_IT_trconv3_bi': 'IT.conv3 cluster 5',
    # 'CORnet-S_cluster4_v2_IT_trconv3_bi': 'IT.conv3 cluster 4',
    # 'CORnet-S_cluster3_v2_IT_trconv3_bi': 'IT.conv3 cluster 3',
    # 'CORnet-S_cluster6_v2_IT_trconv3_bi': 'IT.conv3 cluster 6',
    # 'CORnet-S_cluster7_v2_IT_trconv3_bi': 'IT.conv3 cluster 7',
    # 'CORnet-S_cluster8_v2_IT_trconv3_bi': 'IT.conv3 cluster 8',
    # 'CORnet-S_cluster_v2_IT_trconv3_bi_seed42': 'IT.conv3 cluster',  # 0.4081663359539218
    # 'CORnet-S_train_conv3_bi' : 'Train only conv3s',
    # 'CORnet-S_cluster2_v2_IT_bi_seed42' : '',
    # 'CORnet-S_brain_wmc15_IT_bi' : 'Init no train',
    'CORnet-S_cluster2_v2_IT_trconv3_bi_seed42': 'Gabor + Cluster',
    # 'CORnet-S_cluster2_v2_IT_trconv3_bi' : 'Cluster v2 IT',
    'CORnet-S_cluster10_IT_trconv3_bi': 'All cluster',
    # 'CORnet-S_cluster2_v2_V4_trconv3_bi_seed42' : 'Cluster v2',
    # 'CORnet-S_cluster10_V4_trconv3_bi' : 'All cluster',
    'CORnet-S_cluster11_IT_trconv3_bi': 'Only gabors + KN',
    # full train:
    # 'CORnet-S_train_V4': 'V2 random train after',
    # # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_ln5_wmc6_bi_full':'V2.conv2 train',
    # # 'CORnet-S_train_gabor_dist_both_kernel_bi_BF': 'V1 init train after',
    # # 'CORnet-S_train_gabor_dist_both_kernel_bi_full':'V1 init train full',
    # 'CORnet-S_train_gmk1_gmk2_ln3_kn4_ln5_bi' : 'V2.conv1 init train after',
    # 'CORnet-S_train_gmk1_gmk2_ln3_kn4_ln5_bi_full':'V2.conv1 train full',
    # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_ln5_wmc6_bi' : 'V2.conv2',

    # epoch 43
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_wm6_full' : 'Imagenet optimized until V2.conv2',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_full' : 'Brain benchmark optimized until V2.conv2'

    # Batchnorm corrected
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5': 'L5 no batchnorm',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_BF': 'L5 batchnorm',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_wm6_v2': 'L6 no batchnorm',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_wm6_v2_BF': 'L6 batchnorm',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5': 'L5.2 no batchnorm',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_BF': 'L5.2 batchnorm',

    'CORnet-S_full': 'Standard train',
}

if __name__ == '__main__':
    # plot_benchmarks_over_epochs('CORnet-S_full',
    #                             (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 5, 7, 10, 15, 20, 43))
    # plot_figure_2()
    plot_models_benchmarks(models, 'first_generation', benchmarks)
    # plot_model_avg_benchmarks(models, 'first_generation')
    # image_scores('CORnet-S_brainboth2_IT_trconv3_bi', 'CORnet-S_full',[100,1000,10000,100000,500000], brain=True)
    # image_scores('CORnet-S_cluster2_v2_IT_trconv3_bi', 'CORnet-S_full', [100, 1000, 10000, 100000, 500000], brain=True)
    # image_scores('CORnet-S_cluster2_v2_IT_trconv3_bi', 'CORnet-S_full', [100, 1000, 10000, 100000, 500000], brain=False)
    # plot_single_benchmarks({
    #                         # 'CORnet-S_train_gabor_dist_both_kernel_bi_BF': 'V1',
    #                         # 'CORnet-S_train_gabor_dist_both_kernel_bi_full': 'V1 train',
    #                         'CORnet-S_train_gmk1_gmk2_ln3_kn4_ln5_bi':'V2.conv1 init train after',
    #                         'CORnet-S_train_gmk1_gmk2_ln3_kn4_ln5_bi_full':'V2.conv1 init train full',
    #                         # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_ln5_wmc6_bi_full': 'V2.conv2 train',
    #                         'CORnet-S_full': 'Standard train',
    #                         # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_ln5_wmc6_bi': 'V2.conv2',
    #                         }
    #                        , epochs=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # plot_single_benchmarks({'CORnet-S_train_gabor_dist_both_kernel_bi_BF': 'V1',
    #                         # 'CORnet-S_train_wmk0_wmc1_bi':'V1 kernel dist',
    #                        # 'CORnet-S_train_wmc0_wmc1_bi': 'V1 channel dist',
    #                         'CORnet-S_train_kn1_kn2_bi_v2':'V1 kernel normal',
    #                         # 'CORnet-S_train_kn1_kn2_kn3_ln4_ln5_wmc6_bi' : 'V2.conv2',
    #                         'CORnet-S_full': 'Full',
    #                           }
    #                        , epochs=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], run_mean=False)
    # plot_single_benchmarks({'CORnet-S_train_gabor_dist_both_kernel_bi_BF': 'V1',
    #                         'CORnet-S_train_gmk1_gmk2_kn3_ln4_ln5_wmc6_bi' : 'V2.conv2',
    #                         # 'CORnet-S_train_wmk1_wmk2_kn3_ln4_ln5_wmc6_bi':'V2.conv2 kernel dist',
    #                         # 'CORnet-S_train_wmc1_wmc2_kn3_ln4_ln5_wmc6_bi': 'V2.conv2 channel dist',
    #                         'CORnet-S_train_kn1_kn2_kn3_ln4_ln5_wmc6_bi':'V2.conv2 new mixture channel',
    #                         'CORnet-S_full': 'Full',
    #                         }
    #                        , epochs=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    # plot_first_epochs({  # 'CORnet-S_train_kn1_kn2_kn3_kn4_kn5_kn6_bi' : 'V2.conv2',
    #     # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_kn6_kn7train_bi': 'V2.conv3_trained',
    #     # 'CORnet-S_brain_t7_t12_wmc15_IT_bi': 'IT.conv3_trained',
    #     # 'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_tr_bi': 'V4.conv3_trained',
    #     'CORnet-S_cluster2_v2_IT_trconv3_bi': 'IT.conv3_special',
    #     'CORnet-S_cluster2_v2_V4_trconv3_bi_seed42': 'V4.conv3_special',
    #     'CORnet-S_train_gmk1_cl2_7_7tr_bi_seed42': 'V2.conv3_special',
    #     # 'CORnet-S_brain2_t7_t12_knall_IT_bi_v2' : 'IT.conv2',
    #     'CORnet-S_full': 'Standard training',
    # }
    #     , epochs=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 10, 20], convergence=True,
    #     brain=True)
    # plot_first_epochs({  # 'CORnet-S_train_kn1_kn2_kn3_kn4_kn5_kn6_bi' : 'V2.conv2',
    #     # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_kn6_kn7train_bi': 'V2.conv3_trained',
    #     'CORnet-S_brain_t7_t12_wmc15_IT_bi': 'IT.conv3_trained',
    #     'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_tr_bi': 'V4.conv3_trained',
    #     # 'CORnet-S_brain2_t7_t12_knall_IT_bi_v2' : 'IT.conv2',
    #     'CORnet-S_full': 'Standard training',
    # }
    #     , epochs=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6], convergence=True, brain=False)
    plot_first_epochs({  # 'CORnet-S_train_kn1_kn2_kn3_kn4_kn5_kn6_bi' : 'V2.conv2',
        # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_kn6_kn7train_bi': 'V2.conv3_trained',
        # 'CORnet-S_brain_t7_t12_wmc15_IT_bi': 'IT.conv3_trained gaussian mixture',
        # 'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_tr_bi': 'V4.conv3_trained',
        # 'CORnet-S_cluster_IT_trconv3_bi' : 'IT.conv3_trained cluster',
        # 'CORnet-S_brain2_t7_t12_knall_IT_bi_v2' : 'IT.conv2',
        'CORnet-S_full': 'Standard training',
        'CORnet-S_full_img500000': 'Standard training 50% imgs',
        'CORnet-S_cluster2_v2_IT_trconv3_bi': 'AG+CT 100% Imgs',
        'CORnet-S_cluster2_v2_IT_trconv3_bi_img500000': 'AG+CT 50% Imgs',
        'CORnet-S_train_conv3_bi_img50000': 'KN+CT 50%',
        'CORnet-S_train_conv3_bi': 'KN+CT 100% Imgs',
    }
        , epochs=[0, 1, 2, 3, 6, 20], convergence=True, brain=True)

    # plot_single_benchmarks(['CORnet-S_full', 'CORnet-S_train_gabor_dist_both_kernel'], brain=False, compare_batchfix=True)
    # # plot_single_benchmarks(best_models_brain, brain=False)
    # score_over_layers(layer_best, layer_random)
    # score_over_layers_avg(best_models_brain_avg_all, random_scores)
    # score_over_layers_avg(best_models_brain_avg_all, random_scores, imagenet=True)
    # score_over_layers_avg(best_special_brain, random_scores)

    # score_over_layers_avg(best_special_brain, random_scores, imagenet=True, convergence=False)
    # [, 'Consecutive Training']
    # plot_models_vs({'Mixture gaussian' : { 'Selective':'CORnet-S_brain_t7_t12_wmc15_IT_bi','Consecutive':'CORnet-S_brain_wmc15_IT_bi'},
    #                 'Cluster' : { 'Selective':'CORnet-S_cluster2_v2_IT_trconv3_bi_seed42', 'Consecutive':'CORnet-S_cluster2_v2_IT_bi_seed42'},
    #                 'Kernel normal' : { 'Selective':'CORnet-S_brain3_t7_t12_knall_IT_bi', 'Consecutive':'CORnet-S_brain3_knall_IT_bi'},
    #                 'No gabor prior' : { 'Selective':'CORnet-S_brain2_t7_t12_knall_IT_bi_v2', 'Consecutive':'CORnet-S_brain2_knall_IT_bi_v2'},
    #                 }, 'comparison', convergence=False)
    # plot_models_vs({'Mixture gaussian' : { 'Selective':'CORnet-S_brain_t7_t12_wmc15_IT_bi','Consecutive':'CORnet-S_brain_wmc15_IT_bi'},
    #                 'Cluster' : { 'Selective':'CORnet-S_cluster2_v2_IT_trconv3_bi_seed42', 'Consecutive':'CORnet-S_cluster2_v2_IT_bi_seed42'},
    #                 'Kernel normal' : { 'Selective':'CORnet-S_brain3_t7_t12_knall_IT_bi', 'Consecutive':'CORnet-S_brain3_knall_IT_bi'},
    #                 'No gabor prior' : { 'Selective':'CORnet-S_brain2_t7_t12_knall_IT_bi_v2', 'Consecutive':'CORnet-S_brain2_knall_IT_bi_v2'},
    #                 }, 'comparison', convergence=False, imagenet=True)
