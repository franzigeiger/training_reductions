import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from benchmark.database import load_scores, get_connection
from nets.global_data import layer_best_2, random_scores, best_special_brain, layers, convergence_epoch, benchmarks, \
    benchmarks_public, best_models_brain_avg_all, convergence_images
from plot.plot_data import plot_data_base, plot_bar_benchmarks, scatter_plot, plot_data_double


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


def plot_models_benchmarks(models, file_name):
    model_dict = {}
    conn = get_connection()
    epoch = 6
    names = []
    for model in models.keys():
        names.append(f'{model}_epoch_{epoch:02d}')
    model_dict = load_scores(conn, names, benchmarks)
    benchmarks_labels = ['V1', 'V2', 'V4', 'IT', 'Behavior', 'Imagenet']
    data_set = {}
    # We replace the model id, a more human readable version
    for id, desc in models.items():
        data_set[desc] = model_dict[f'{id}_epoch_{epoch:02d}']
        print(f'Mean of brain benchmark model {desc}, {np.mean(data_set[desc][2:5])}')
    plot_bar_benchmarks(data_set, benchmarks_labels, 'Model scores in epoch 6', 'Scores', file_name)


def plot_model_avg_benchmarks(models, file_name):
    model_dict = {}
    conn = get_connection()
    epoch = 6
    names = []
    for model in models.keys():
        names.append(f'{model}_epoch_{epoch:02d}')
    model_dict = load_scores(conn, names, benchmarks)
    benchmarks_labels = ['mean(IT,V4,Behavior)', 'Imagenet']
    data_set = {}
    # We replace the model id, a more human readable version
    for id, desc in models.items():
        data = model_dict[f'{id}_epoch_{epoch:02d}']
        data_set[desc] = [np.mean(data[2:5]), data[5]]
        print(f'Mean of brain benchmark model {desc}, {np.mean(data[2:5])}')
    plot_bar_benchmarks(data_set, benchmarks_labels, 'Model scores in epoch 6', 'Scores', file_name)


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
    plot_bar_benchmarks(data_set, benchmarks_labels, 'Model scores in epoch 6', 'Scores', file_name)


def plot_benchmarks_over_epochs(model, epochs=None):
    benchmarks = [
        # 'movshon.FreemanZiemba2013.V1-pls',
        #                        'movshon.FreemanZiemba2013.V2-pls',
        'dicarlo.Majaj2015.V4-pls',
        'dicarlo.Majaj2015.IT-pls',
        'dicarlo.Rajalingham2018-i2n',
        'fei-fei.Deng2009-top1']
    model_dict = {}
    conn = get_connection()
    if epochs is None:
        epochs = (0, 5, 10, 15, 20)

    names = []
    for epoch in epochs:
        if epoch % 1 == 0:
            names.append(f'{model}_epoch_{epoch:02d}')
        else:
            names.append(f'{model}_epoch_{epoch:.1f}')
    model_dict[model] = load_scores(conn, names, benchmarks)
    # model_dict[model][f'{model}_epoch_00'] = load_scores(conn, ['CORnet-S_random'], benchmarks)['CORnet-S_random']
    full = model_dict[model][f'{model}_epoch_43']
    benchmarks_labels = ['V4', 'IT', 'Behavior', 'Imagenet']  # ['V1', 'V2',]
    data = {bench: [] for bench in benchmarks_labels}
    for i in range(4):
        for epoch in epochs:
            if epoch % 1 == 0:
                frac = (model_dict[model][f'{model}_epoch_{epoch:02d}'][i] / full[i]) * 100
            else:
                frac = (model_dict[model][f'{model}_epoch_{epoch:.1f}'][i] / full[i]) * 100
            data[benchmarks_labels[i]].append(frac)
        # data['CORnet-S_full_epoch_0'] = [0]*3 + [model_dict['CORnet-S']['CORnet-S'][i]]
    # plot_data_base(data, f'Resnet50(Michael version) Brain-Score benchmarks', epochs, 'Epoch', 'Score', x_ticks=epochs,
    #                log=True)
    plot_data_base(data, f'CORnet-S Brain-Score benchmarks', epochs, 'Epoch', 'Score [% of  final score]',
                   x_ticks=[value for value in epochs if value not in [0.1, 0.3, 0.5, 0.7, 0.9]], percent=True,
                   log=True, annotate=True, annotate_pos=10)
    # plot_data_base(data, f'{model} Brain-Score benchmarks', epochs, 'Epoch', 'Score', x_ticks=epochs)


def plot_first_epochs(models, epochs=None, brain=True, convergence=True):
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
    plot_data_base(data, title, x_values, 'Epochs', 'Score', x_ticks=epochs + [10, 20, 30, 40, 50], log=True,
                   percent=True, special_xaxis=True)


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


def score_over_layers(models, random):
    conn = get_connection()
    names = []
    for model in models.keys():
        if model != "CORnet-S_random":
            names.append(f'{model}_epoch_06')
        else:
            names.append(model)
    model_dict = load_scores(conn, names, benchmarks)
    benchmarks_labels = ['V1', 'V2', 'V4', 'IT', 'Behavior', 'Imagenet']
    data = {}
    for i in range(6):
        data[benchmarks_labels[i]] = []
        for model, number in models.items():
            if model != "CORnet-S_random":
                data[benchmarks_labels[i]].append(model_dict[f'{model}_epoch_06'][i])
            else:
                data[benchmarks_labels[i]].append(model_dict[f'{model}'][i])

    for model in random.keys():
        if model != "CORnet-S_random":
            names.append(f'{model}_epoch_06')
        else:
            names.append(model)
    model_dict = load_scores(conn, names, benchmarks)
    data2 = {}
    for i in range(6):
        data2[benchmarks_labels[i]] = []
        for model, number in random.items():
            if model != "CORnet-S_random":
                data2[benchmarks_labels[i]].append(model_dict[f'{model}_epoch_06'][i])
            else:
                data2[benchmarks_labels[i]].append(model_dict[f'{model}'][i])

    plot_data_double(data, data2, f'Benchmarks over layers', x_name='Number of trained layers', y_name='Score',
                     x_ticks=list(models.values()), scale_fix=[0.0, 0.6], x_ticks_2=list(random.values()))


def plot_figure_2():
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    score_over_layers_avg(best_special_brain, best_models_brain_avg_all, random_scores, imagenet=False,
                          convergence=False, ax=ax1)
    score_over_layers_avg(best_special_brain, best_models_brain_avg_all, random_scores, imagenet=True,
                          convergence=False, ax=ax2)
    # file_name = name.replace(' ', '_')
    plt.savefig(f'figure_2.png')
    plt.show()


def score_over_layers_avg(models, second, random, imagenet=False, convergence=False, ax=None):
    conn = get_connection()
    names = []

    for model in list(models.keys()) + list(second.keys()):
        if convergence and model in convergence_epoch:
            postfix = f'_epoch_{convergence_epoch[model]:02d}'
        else:
            postfix = f'_epoch_06'
        if model != "CORnet-S_random":
            names.append(f'{model}{postfix}')
        else:
            names.append(model)
    # model_dict = load_scores(conn, names, benchmarks)
    data = {}

    full = 0
    for model in random.keys():
        if model != "CORnet-S_random":
            if convergence:
                names.append(f'{model}_epoch_{convergence_epoch[model]:02d}')
            else:
                names.append(f'{model}_epoch_06')
        else:
            names.append(model)
    model_dict = load_scores(conn, names, benchmarks)
    data2 = {}
    data2['Score'] = []
    layers2 = []
    for model, layer in random.items():
        if convergence and model in convergence_epoch:
            postfix = f'_epoch_{convergence_epoch[model]:02d}'
        else:
            postfix = f'_epoch_06'
        layers2.append(layer_best_2[layer])
        if imagenet:
            if model == "CORnet-S_random":
                percent = (np.mean(model_dict[f'{model}'][5]) / full) * 100
                data2['Score'].append(percent)
            elif model == 'CORnet-S_full':
                full = np.mean(model_dict[f'{model}{postfix}'][5])
                data2['Score'].append(100)
            else:
                percent = (np.mean(model_dict[f'{model}{postfix}'][5]) / full) * 100
                data2['Score'].append(percent)
        else:
            if model == "CORnet-S_random":
                percent = (np.mean(model_dict[f'{model}'][2:5]) / full) * 100
                data2['Score'].append(percent)
            elif model == 'CORnet-S_full':
                full = np.mean(model_dict[f'{model}{postfix}'][2:5])
                data2['Score'].append(100)
            else:
                percent = (np.mean(model_dict[f'{model}{postfix}'][2:5]) / full) * 100
                data2['Score'].append(percent)
    x_ticks = {}
    labels = {}
    for models, name in zip([models, second], ['Selective training', 'Consecutive Training']):
        data[name] = []
        layers = []
        for model, layer in models.items():
            if convergence and model in convergence_epoch:
                postfix = f'_epoch_{convergence_epoch[model]:02d}'
            else:
                postfix = f'_epoch_06'
            layers.append(layer_best_2[layer])
            if imagenet:
                if model == "CORnet-S_random":
                    percent = (np.mean(model_dict[f'{model}'][5]) / full) * 100
                    data[name].append(percent)
                elif model == 'CORnet-S_full':
                    full = np.mean(model_dict[f'{model}{postfix}'][5])
                    data[name].append(100)
                else:
                    percent = (np.mean(model_dict[f'{model}{postfix}'][5]) / full) * 100
                    data[name].append(percent)
            else:
                if model == "CORnet-S_random":
                    percent = (np.mean(model_dict[f'{model}'][2:5]) / full) * 100
                    data[name].append(percent)
                elif model == 'CORnet-S_full':
                    full = np.mean(model_dict[f'{model}{postfix}'][2:5])
                    data[name].append(100)
                else:
                    percent = (np.mean(model_dict[f'{model}{postfix}'][2:5]) / full) * 100
                    data[name].append(percent)
        x_ticks[name] = layers
        labels[name] = [value.replace('special', 'trained') if 'special' in value else value for value in
                        models.values()]

    if imagenet:
        title = f'Imagenet over layers'
        y = 'Imagenet[% of standard training]'
    else:
        title = f'Brain-Score Benchmark mean(V4, IT, Behavior) over layers'
        y = 'mean(V4, IT, Behavior)[% of standard training]'

    plot_data_double(data, data2, '', x_name='Number of trained layers',
                     y_name=y, x_ticks=x_ticks,
                     x_ticks_2=layers2, percent=True, data_labels=labels, ax=ax)


def image_scores(model1, model2, imgs, brain=True, ax=None):
    names = []
    conn = get_connection()
    for model in [model1, model2]:
        for img in imgs:
            name = f'{model}_img{img}'
            names.append(f'{name}_epoch_{convergence_images[name]}')
    model_dict = load_scores(conn, names, benchmarks)
    data = {'Selective training': []}
    data2 = {'Score': []}
    for i in imgs:
        name1 = f'{model1}_img{i}'
        name2 = f'{model2}_img{i}'
        if brain:
            data['Selective training'].append(np.mean(model_dict[f'{name1}_epoch_{convergence_images[name1]}'][2:4]))
            data2['Score'].append(np.mean(model_dict[f'{name2}_epoch_{convergence_images[name2]}'][2:4]))
        else:
            data['Selective training'].append(np.mean(model_dict[f'{name1}_epoch_{convergence_images[name1]}'][5]))
            data2['Score'].append(np.mean(model_dict[f'{name2}_epoch_{convergence_images[name2]}'][5]))
    if brain:
        title = f'Brain scores mean vs number of weights'
        y = 'Score'
    else:
        title = 'Imagenet score vs number of weights'
        y = 'Score'
    plot_data_double(data, data2, title, x_name='Number of images', y_name=y, x_ticks={'Selective training': imgs},
                     x_ticks_2=imgs, percent=False, log=True, ax=ax)


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
                 title=title, ax=ax)


models = {
    # Batchnrom corrected

    # #  Layer 1 & 2
    # 'CORnet-S_train_V2': 'V1 random train after',
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
    # 'CORnet-S_train_cn1_cn2_bi' : 'V1 channel normal',
    # 'CORnet-S_train_gmk1_bd2_bi' : 'Gabor plus best dist',
    # 'CORnet-S_train_gmk1_cl2_bi' : 'Cluster l2',

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
    'CORnet-S_cluster2_v2_IT_trconv3_bi': 'IT.conv3 Cluster',
    'CORnet-S_cluster5_v2_IT_trconv3_bi': 'IT.conv3 cluster 5',
    'CORnet-S_cluster4_v2_IT_trconv3_bi': 'IT.conv3 cluster 4',
    'CORnet-S_cluster3_v2_IT_trconv3_bi': 'IT.conv3 cluster 3',
    'CORnet-S_cluster6_v2_IT_trconv3_bi': 'IT.conv3 cluster 6',
    'CORnet-S_cluster7_v2_IT_trconv3_bi': 'IT.conv3 cluster 7',
    'CORnet-S_cluster8_v2_IT_trconv3_bi': 'IT.conv3 cluster 8',
    'CORnet-S_cluster_v2_IT_trconv3_bi': 'IT.conv3 cluster',  # 0.4081663359539218
    # 'CORnet-S_train_conv3_bi' : 'Train only conv3s',
    # 'CORnet-S_brain_wmc15_IT_bi' : 'Init no train',
    # 'CORnet-S_cluster_v2_IT_trconv3_bi' : 'Cluster v2',

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
    # plot_models_benchmarks(models, 'first_generation')
    # plot_model_avg_benchmarks(models, 'first_generation')
    # image_scores('CORnet-S_brainboth2_IT_trconv3_bi', 'CORnet-S_full',[100,1000,10000,100000,500000], brain=True)
    image_scores('CORnet-S_brain_t7_t12_wmc15_IT_bi', 'CORnet-S_full', [100, 1000, 10000, 100000, 500000], brain=True)
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
    #     'CORnet-S_brain_t7_t12_wmc15_IT_bi': 'IT.conv3_trained',
    #     'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_tr_bi': 'V4.conv3_trained',
    #     # 'CORnet-S_brain2_t7_t12_knall_IT_bi_v2' : 'IT.conv2',
    #     'CORnet-S_full': 'Standard training',
    # }
    #     , epochs=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6], convergence=True, brain=True)
    # plot_first_epochs({  # 'CORnet-S_train_kn1_kn2_kn3_kn4_kn5_kn6_bi' : 'V2.conv2',
    #     # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_kn6_kn7train_bi': 'V2.conv3_trained',
    #     'CORnet-S_brain_t7_t12_wmc15_IT_bi': 'IT.conv3_trained',
    #     'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_tr_bi': 'V4.conv3_trained',
    #     # 'CORnet-S_brain2_t7_t12_knall_IT_bi_v2' : 'IT.conv2',
    #     'CORnet-S_full': 'Standard training',
    # }
    #     , epochs=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6], convergence=True, brain=False)
    # plot_first_epochs({  # 'CORnet-S_train_kn1_kn2_kn3_kn4_kn5_kn6_bi' : 'V2.conv2',
    #     # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_kn6_kn7train_bi': 'V2.conv3_trained',
    #     'CORnet-S_brain_t7_t12_wmc15_IT_bi': 'IT.conv3_trained gaussian mixture',
    #     # 'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_tr_bi': 'V4.conv3_trained',
    #     'CORnet-S_cluster_IT_trconv3_bi' : 'IT.conv3_trained cluster',
    #     # 'CORnet-S_brain2_t7_t12_knall_IT_bi_v2' : 'IT.conv2',
    #     'CORnet-S_full': 'Standard training',
    # }
    #     , epochs=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6], convergence=True, brain=False)

    # plot_single_benchmarks(['CORnet-S_full', 'CORnet-S_train_gabor_dist_both_kernel'], brain=False, compare_batchfix=True)
    # # plot_single_benchmarks(best_models_brain, brain=False)
    # score_over_layers(layer_best, layer_random)
    # score_over_layers_avg(best_models_brain_avg_all, random_scores)
    # score_over_layers_avg(best_models_brain_avg_all, random_scores, imagenet=True)
    # score_over_layers_avg(best_special_brain, random_scores)

    # score_over_layers_avg(best_special_brain, random_scores, imagenet=True, convergence=False)
