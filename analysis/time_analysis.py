import numpy as np

from benchmark.database import load_scores
from plot.plot_data import get_connection, plot_data_base, plot_bar_benchmarks, scatter_plot

layers = ['full', 'V1.conv1', 'V1.conv2',
          'V2.conv_input', 'V2.skip', 'V2.conv1', 'V2.conv2', 'V2.conv3',
          'V4.conv_input', 'V4.skip', 'V4.conv1', 'V4.conv2', 'V4.conv3',
          'IT.conv_input', 'IT.skip', 'IT.conv1', 'IT.conv2', 'IT.conv3', 'decoder']


def plot_over_epoch(models):
    model_dict = {}
    conn = get_connection()
    epochs = (0, 5, 10, 15)
    for model in models:
        names = []
        for epoch in epochs:
            names.append(f'{model}_epoch_{epoch:02d}')
        model_dict[model] = load_scores(conn, names,
                                        ['movshon.FreemanZiemba2013.V1-pls',
                                         'movshon.FreemanZiemba2013.V2-pls',
                                         'dicarlo.Majaj2015.V4-pls',
                                         'dicarlo.Majaj2015.IT-pls',
                                         'dicarlo.Rajalingham2018-i2n',
                                         'fei-fei.Deng2009-top1'])
    model_dict[f'{model}_epoch_00'] = load_scores(conn, ['CORnet-S_random'],
                                                  ['movshon.FreemanZiemba2013.V1-pls',
                                                   'movshon.FreemanZiemba2013.V2-pls',
                                                   'dicarlo.Majaj2015.V4-pls',
                                                   'dicarlo.Majaj2015.IT-pls',
                                                   'dicarlo.Rajalingham2018-i2n',
                                                   'fei-fei.Deng2009-top1'])
    benchmarks = ['V1', 'V2', 'V4', 'IT', 'Behavior', 'Imagenet']
    for i in range(6):
        data = {}
        for model in models:
            data[model] = []
            for epoch in epochs:
                data[model].append(model_dict[model][f'{model}_epoch_{epoch:02d}'][i])
        # data['CORnet-S'] = [0] * 3 + [model_dict['CORnet-S']['CORnet-S'][i]]
        plot_data_base(data, f'{benchmarks[i]} Benchmark over epochs', epochs, 'Score over epochs', 'Score')


def plot_models_benchmarks(models, file_name):
    model_dict = {}
    conn = get_connection()
    epoch = 6
    names = []
    for model in models.keys():
        names.append(f'{model}_epoch_{epoch:02d}')
    model_dict = load_scores(conn, names,
                             ['movshon.FreemanZiemba2013.V1-pls',
                              'movshon.FreemanZiemba2013.V2-pls',
                              'dicarlo.Majaj2015.V4-pls',
                              'dicarlo.Majaj2015.IT-pls',
                              'dicarlo.Rajalingham2018-i2n'
                                 , 'fei-fei.Deng2009-top1'
                              ])
    benchmarks = ['V1', 'V2', 'V4', 'IT', 'Behavior', 'Imagenet']
    data_set = {}
    # We replace the model id, a more human readable version
    for id, desc in models.items():
        data_set[desc] = model_dict[f'{id}_epoch_{epoch:02d}']
        print(f'Mean of brain benchmark model {desc}, {np.mean(data_set[desc][2:5])}')
    plot_bar_benchmarks(data_set, benchmarks, 'Model scores in epoch 5', 'Scores', file_name)


def plot_benchmarks_over_epochs(model, epochs=None):
    model_dict = {}
    conn = get_connection()
    if epochs is None:
        epochs = (0, 5, 10, 15, 20)

    names = []
    for epoch in epochs:
        names.append(f'{model}_epoch_{epoch:02d}')
    model_dict[model] = load_scores(conn, names,
                                    ['movshon.FreemanZiemba2013.V1-pls',
                                     'movshon.FreemanZiemba2013.V2-pls',
                                     'dicarlo.Majaj2015.V4-pls',
                                     'dicarlo.Majaj2015.IT-pls',
                                     'dicarlo.Rajalingham2018-i2n',
                                     'fei-fei.Deng2009-top1'])
    model_dict[model][f'{model}_epoch_00'] = load_scores(conn, ['CORnet-S_random'],
                                                         ['movshon.FreemanZiemba2013.V1-pls',
                                                          'movshon.FreemanZiemba2013.V2-pls',
                                                          'dicarlo.Majaj2015.V4-pls',
                                                          'dicarlo.Majaj2015.IT-pls',
                                                          'dicarlo.Rajalingham2018-i2n',
                                                          'fei-fei.Deng2009-top1'])['CORnet-S_random']
    benchmarks = ['V1', 'V2', 'V4', 'IT', 'Behavior', 'Imagenet']
    data = {bench: [] for bench in benchmarks}
    for i in range(6):
        for epoch in epochs:
            data[benchmarks[i]].append(model_dict[model][f'{model}_epoch_{epoch:02d}'][i])
        # data['CORnet-S_full_epoch_0'] = [0]*3 + [model_dict['CORnet-S']['CORnet-S'][i]]
    plot_data_base(data, f'Resnet50(Michael version) Brain-Score benchmarks', epochs, 'Epoch', 'Score', x_ticks=epochs,
                   log=True)
    # plot_data_base(data, f'CORnet-S(Franzi version) Brain-Score benchmarks', epochs, 'Epoch', 'Score', x_ticks=epochs)
    # plot_data_base(data, f'{model} Brain-Score benchmarks', epochs, 'Epoch', 'Score', x_ticks=epochs)


def plot_first_epochs(models, epochs=None, brain=True):
    model_dict = {}
    conn = get_connection()
    if epochs is None:
        epochs = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6)
    data = {}
    for model, name in models.items():
        names = []
        for epoch in epochs:
            if epoch % 1 == 0:
                names.append(f'{model}_epoch_{epoch:02d}')
            else:
                names.append(f'{model}_epoch_{epoch:.1f}')
        model_dict = load_scores(conn, names,
                                 ['movshon.FreemanZiemba2013.V1-pls',
                                  'movshon.FreemanZiemba2013.V2-pls',
                                  'dicarlo.Majaj2015.V4-pls',
                                  'dicarlo.Majaj2015.IT-pls',
                                  'dicarlo.Rajalingham2018-i2n',
                                  'fei-fei.Deng2009-top1'])
        scores = []
        for epoch in epochs:
            if brain:
                if epoch % 1 == 0:
                    scores.append(np.mean(model_dict[f'{model}_epoch_{int(epoch):02d}'][2:5]))
                else:
                    scores.append(np.mean(model_dict[f'{model}_epoch_{epoch:.1f}'][2:5]))
            else:
                if epoch % 1 == 0:
                    scores.append(model_dict[f'{model}_epoch_{int(epoch):02d}'][5])
                else:
                    scores.append(model_dict[f'{model}_epoch_{epoch:.1f}'][5])
        data[name] = scores
    title = f'Brain scores mean vs epochs' if brain else 'Imagenet score vs epochs'
    plot_data_base(data, title, epochs, 'Epochs', 'Score', x_ticks=epochs, log=True)


def plot_single_benchmarks(models, epochs=None, compare_batchfix=False):
    model_dict = {}
    conn = get_connection()
    if epochs is None:
        epochs = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6)
    data = {}
    benchmarks = ['V1', 'V2', 'V4', 'IT', 'Behavior', 'Imagenet']

    for model, name in models.items():
        names = []
        for epoch in epochs:
            if epoch % 1 == 0:
                names.append(f'{model}_epoch_{epoch:02d}')
            else:
                names.append(f'{model}_epoch_{epoch:.1f}')

            if compare_batchfix:
                names.append(f'{model}_epoch_{epoch:02d}_BF')
        model_dict[name] = load_scores(conn, names,
                                       ['movshon.FreemanZiemba2013.V1-pls',
                                        'movshon.FreemanZiemba2013.V2-pls',
                                        'dicarlo.Majaj2015.V4-pls',
                                        'dicarlo.Majaj2015.IT-pls',
                                        'dicarlo.Rajalingham2018-i2n',
                                        'fei-fei.Deng2009-top1'])
    for i in range(6):
        for model, name in models.items():
            scores = []
            for epoch in epochs:
                if epoch % 1 == 0:
                    scores.append(model_dict[name][f'{model}_epoch_{int(epoch):02d}'][i])
                else:
                    scores.append(model_dict[name][f'{model}_epoch_{epoch:.1f}'][i])

            data[name] = scores
            if compare_batchfix:
                scores = []
                for epoch in epochs:
                    scores.append(model_dict[name][f'{model}_BF_epoch_{epoch:02d}'][i])
                data[f'{name}_BF'] = scores

        title = f'{benchmarks[i]} benchmark vs epochs'
        plot_data_base(data, title, epochs, 'Epoch', 'Score', x_ticks=epochs)


best_models_imagenet = {
    'CORnet-S_full': 'full',
    'CORnet-S_train_gabor_multi_dist': 'V1.conv1',
    'CORnet-S_train_gabor_dist_weight_dist_channel': 'V1.conv2',
    'CORnet-S_train_gmk1_wmc2_kn3': 'V2.conv_input',
    'CORnet-S_train_gmk1_wmc2_kn3_kn4': 'V2.skip',
    'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5': 'V2.conv1',
    'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_wm6_v2': 'V2.conv2',
    'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wmc6_kn7_v2': 'V2.conv3',
    'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12': 'V4.conv2',
}
best_models_brain = {
    'CORnet-S_full': 'full',
    'CORnet-S_train_gabor_multi_dist': 'V1.conv1',
    'CORnet-S_train_gabor_dist_weight_dist_kernel': 'V1.conv2',
    'CORnet-S_train_gmk1_gmk2_kn3_kn4': 'V2.skip',
    'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5': 'V2.conv1',
    'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_1_gpu': 'V2.conv2',
    'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_bdk7': 'V2.conv3',
    'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12': 'V4.conv2',
}

best_models_brain_2 = {
    'CORnet-S_train_gabor_dist_both_kernel_bi_BF': 'V1',
    # 'CORnet-S_train_gmk1_gmk2_ln3_bi': 'V2.input',
    # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_bi': 'V2.skip',
    'CORnet-S_train_gmk1_gmk2_ln3_kn4_ln5_bi': 'V2.conv1',
    'CORnet-S_train_gmk1_gmk2_kn3_ln4_ln5_wmc6_bi': 'V2.conv2',
    'CORnet-S_full': 'Full',
}

random_scores = {
    'CORnet-S_train_V2': 'V1.conv2',
    'CORnet-S_train_V4': 'V2.conv3',
    'CORnet-S_train_IT_seed_0': 'V4.conv3',
    "CORnet-S_random": 'decoder',
    'CORnet-S_train_random': 'IT.conv3'
}


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

    #  Layer 1 & 2
    'CORnet-S_train_V2': 'Random init',
    # 'CORnet-S_train_gabor_dist_weight_dist_channel' : 'V1 base',
    # 'CORnet-S_train_gabor_dist_weight_dist_channel_ra_BF' : 'V1 no batchnorm',
    # 'CORnet-S_train_gabor_dist_weight_dist_channel_bi_BF' : 'V1 batchnorm',
    # 'CORnet-S_train_gabor_dist_both_kernel' : "V1 gabor ",
    # 'CORnet-S_train_gabor_dist_both_kernel_ra_BF' : "V1 gabor no batchnorm",
    # 'CORnet-S_train_gabor_dist_both_kernel_bi_BF' : "V1 gabor batchnorm", # best

    'CORnet-S_train_gabor_dist_both_kernel_bi_BF': 'V1',
    'CORnet-S_train_wmk0_wmc1_bi': 'V1 kernel dist',
    # 'CORnet-S_train_wmc0_wmc1_bi': 'V1 channel dist',
    'CORnet-S_train_kn1_kn2_bi': 'V1 kernel normal',
    'CORnet-S_train_ln1_ln2_bi': 'V1 layer normal',

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

    # # # Layer 6
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_bi': 'Best layer 5',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_ln5_wm6_ra': 'V2.conv2 weight dist',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_ln5_wm6_bi': 'V2.conv2 weight dist batchnorm',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_bi': 'V2.conv2 weight dist batchnorm V2',
    # 'CORnet-S_train_gmk1_gmk2_ln3_ln4_ln5_wm6_ra': 'V2.conv2 weight dist V2',
    # 'CORnet-S_train_gmk1_gmk2_ln3_ln4_ln5_wm6_bi' : 'V2.conv2 weight dist V3 batchnorm',
    # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_kn5_wmc6_bi' : 'V2.conv2 weight dist V4 batchnorm',
    # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_ln5_wmc6_bi' : 'V2.conv2',

    # 'CORnet-S_train_gabor_dist_both_kernel_bi_BF': 'V1',
    # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_ln5_wmc6_bi' : 'V2.conv2',
    # 'CORnet-S_train_wmk1_wmk2_kn3_ln4_ln5_wmc6_bi':'V2.conv2 kernel dist',
    # 'CORnet-S_train_wmc1_wmc2_kn3_ln4_ln5_wmc6_bi': 'V2.conv2 channel dist',
    # 'CORnet-S_train_kn1_kn2_kn3_ln4_ln5_wmc6_bi':'V2.conv2 1',
    # 'CORnet-S_train_kn1_kn2_kn3_ln4_ln5_wmk6_bi':'V2.conv2 2',
    # 'CORnet-S_train_kn1_kn2_kn3_kn4_kn5_kn6_bi' : 'V2.conv2 3',

    # layer 7
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_kn7_bi' : 'Layer 7 bn',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wmk6_kn7_bi' : 'Layer 7 wk bn',
    # 'CORnet-S_brain_kn8_kn9_kn10_wmk11_kn12_bi' : 'Layer 12  V1 bn',
    # 'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_bi' : 'Layer 12 V2  bn',

    # full train:
    # 'CORnet-S_train_gabor_dist_both_kernel_bi_full':'V1 train',
    # 'CORnet-S_train_gmk1_gmk2_ln3_kn4_ln5_bi_full':'V2.conv1 train',
    # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_ln5_wmc6_bi_full':'V2.conv2 train',
    # 'CORnet-S_train_gabor_dist_both_kernel_bi_BF': 'V1',
    # 'CORnet-S_train_gmk1_gmk2_ln3_kn4_ln5_bi' : 'V2.conv1',
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

    'CORnet-S_full': 'Train all',
}

if __name__ == '__main__':
    # plot_benchmarks_over_epochs('CORnet-S_train_IT_seed_0', (5, 10, 15, 20))
    # plot_benchmarks_over_epochs('resnet_mil_trained', (0, 1, 2, 3, 4, 5, 10, 15, 20, 90))
    # plot_benchmarks_over_epochs('CORnet-S_train_all')
    # plot_benchmarks_over_epochs('CORnet-S_full', (0, 1, 2, 3, 5, 7, 10, 15, 20, 43))
    # plot_benchmarks_over_epochs('CORnet-S_train_gabor', (1, 3, 5, 10, 15))
    # plot_benchmarks_over_epochs('CORnet-S_train_gabor_reshape', (1, 3, 5, 10, 15))
    # plot_benchmarks_over_epochs('CORnet-S_train_second', (1, 2, 3, 5, 7))
    # plot_benchmarks_over_epochs('CORnet-S_train_second_no_batchnorm', (1, 2, 3, 5, 10, 15))
    # plot_over_epochs(['CORnet-S_train_all', 'CORnet-S_full'])
    # plot_over_epochs(['CORnet-S_train_all', 'CORnet-S_train_IT_seed_0', 'CORnet-S_gabor', 'CORnet-S_gabor_reshape'])
    plot_models_benchmarks(models, 'first_generation')
    # score_layer_depth(best_models_brain, brain=True)
    # score_layer_depth(best_models_brain, brain=False)

    # score_layer_depth(best_models_imagenet, brain=False)
    # score_layer_depth(best_models_brain, brain=True)

    # plot_first_epochs(best_models_brain, brain=True)
    # plot_first_epochs(best_models_brain_2, brain=True)

    # plot_single_benchmarks(best_models_imagenet, brain=True)
    # , 'CORnet-S_train_gabor_dist_both_kernel': 'V1.conv2'
    # plot_single_benchmarks({'CORnet-S_train_gabor_dist_weight_dist_channel': 'V1'}, compare_batchfix=True)
    # plot_single_benchmarks({
    #     'CORnet-S_train_gabor_dist_both_kernel_bi_BF': 'V1',
    #     'CORnet-S_train_gmk1_gmk2_ln3_bi': 'V2.input',
    #     'CORnet-S_train_gmk1_gmk2_kn3_ln4_bi': 'V2.skip',
    #     'CORnet-S_train_gmk1_gmk2_ln3_kn4_ln5_bi' : 'V2.conv1',
    #     'CORnet-S_train_gmk1_gmk2_kn3_ln4_ln5_wmc6_bi' : 'V2.conv2',
    #     'CORnet-S_full': 'Full',
    # })
    # plot_single_benchmarks({
    #     'CORnet-S_train_gmk1_gmk2_kn3_kn4_ln5_wm6_bi_mom': 'Momentum corrected',
    #     'CORnet-S_train_gmk1_gmk2_kn3_kn4_ln5_wm6_bi': 'Not corrected',
    #     'CORnet-S_full': 'Full',
    # })
    # plot_single_benchmarks(best_models_brain_2, epochs=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    # plot_single_benchmarks(best_models_brain_2, epochs=[0,1,2,3,4,5,6])
    # experiments:
    # plot_single_benchmarks({ 'CORnet-S_train_gabor_dist_both_kernel_bi_BF': 'V1',
    #                        'CORnet-S_train_gabor_dist_both_kernel_bi_full':'V1 train',
    #                        # 'CORnet-S_train_gmk1_gmk2_ln3_kn4_ln5_bi_full':'V2.conv1 train',
    #                        'CORnet-S_train_gmk1_gmk2_kn3_ln4_ln5_wmc6_bi_full':'V2.conv2 train',
    #                          'CORnet-S_full': 'Full',
    #                          'CORnet-S_train_gmk1_gmk2_kn3_ln4_ln5_wmc6_bi':'V2.conv2',
    #                          }
    #                        , epochs=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # plot_single_benchmarks({'CORnet-S_train_gabor_dist_both_kernel_bi_BF': 'V1',
    #                         # 'CORnet-S_train_wmk0_wmc1_bi':'V1 kernel dist',
    #                        # 'CORnet-S_train_wmc0_wmc1_bi': 'V1 channel dist',
    #                         'CORnet-S_train_kn1_kn2_bi':'V1 kernel normal',
    #                         'CORnet-S_train_kn1_kn2_kn3_ln4_ln5_wmc6_bi' : 'V2.conv2',
    #                         'CORnet-S_full': 'Full',
    #                           }
    #                        , epochs=[0,1,2,3,4,5,6])
    # plot_single_benchmarks({'CORnet-S_train_gabor_dist_both_kernel_bi_BF': 'V1',
    #                         'CORnet-S_train_gmk1_gmk2_kn3_ln4_ln5_wmc6_bi' : 'V2.conv2',
    #                         # 'CORnet-S_train_wmk1_wmk2_kn3_ln4_ln5_wmc6_bi':'V2.conv2 kernel dist',
    #                         # 'CORnet-S_train_wmc1_wmc2_kn3_ln4_ln5_wmc6_bi': 'V2.conv2 channel dist',
    #                         'CORnet-S_train_kn1_kn2_kn3_ln4_ln5_wmc6_bi':'V2.conv2 new',
    #                         'CORnet-S_full': 'Full',
    #                         }
    #                        , epochs=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # plot_single_benchmarks(['CORnet-S_full', 'CORnet-S_train_gabor_dist_both_kernel'], brain=False, compare_batchfix=True)
    # plot_single_benchmarks(best_models_brain, brain=False)
