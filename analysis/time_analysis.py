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
    plot_data_base(data, title, epochs, 'Epoch', 'Score', x_ticks=epochs, log=True)


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
        plot_data_base(data, title, epochs, 'Epoch', 'Score', x_ticks=epochs, log=True)


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
    # Base

    # 'CORnet-S_train_IT_seed_0': 'Random init, IT -> train',
    # 'CORnet-S_train_V2': 'Random init',
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    # 'CORnet-S_rand_conv1':'Random init',
    # Beginning
    # 'CORnet-S_train_IT_norm_dist' : 'Normal distributed init, IT -> train',
    # 'CORnet-S_train_V2': 'Random init, V1.conv2-> train',
    # 'CORnet-S_train_gabor': 'Fix gabors, V1.conv2 -> train',
    # 'CORnet-S_train_gabor_reshape':'Fix gabors+normalize, V1.conv2 -> train',
    # 'CORnet-S_train_second': 'Fix gabors and Correlate, V2 -> train',
    # Round 2
    # 'CORnet-S_train_IT_seed_0':'Random init, IT -> train',
    # 'CORnet-S_train_second_corr_only':'Trained Conv1, Correlate Conv2, V2 -> train',
    # 'CORnet-S_train_gabor_fit':'Fit gabor Conv1, V1.Conv2 -> train',
    # 'CORnet-S_train_gabor_fit_second_corr':'Fit gabor Conv1, Correlate Conv2, V2 -> train',
    # Round 3
    # 'CORnet-S_train_second_corr_no_resize' :'Trained Conv1, Correlate Conv2 no resize, V2 -> train',
    # 'CORnet-S_train_second_kernel_conv':'Trained Conv1, Kernel-Conv Conv2, V2 -> train',
    # 'CORnet-S_train_gabor_fit_second_corr_no_resize':'Fit gabor Conv1, Correlate Conv2 no resize, V2 -> train',
    # 'CORnet-S_train_gabor_fit_second_kernel_conv':'Fit gabor Conv1, Kernel-Conv Conv2, V2 -> train',
    # Round 4
    # 'CORnet-S_train_gabor_dist_second_corr_no_resize': 'MultDim gaussian Conv1, Correlate Conv2, V2 -> train',
    # 'CORnet-S_train_gabor_dist' : 'Indep. filter Conv1, V1.Conv2-> train',
    # 'CORnet-S_train_gabor_scrumble' : 'Scrumble fit gabors Con1, V1.Conv2 -> train',
    # 'CORnet-S_train_gabor_multi_dist': 'MG Conv1, V1.Conv2 -> train',
    # 'CORnet-S_train_gabor_dist_second_kernel_conv': 'MultDim gaussian Conv1, Kernel-Conv Conv2, V2 -> train',
    # overview layer 1:
    # 'CORnet-S_rand_conv1':'Random init',
    # # 'CORnet-S_train_gabor': 'Fix gabors',
    # 'CORnet-S_train_gabor_fit': 'Fit gabors',
    # # 'CORnet-S_train_gabor_dist' : 'Gaussian distributions gabor kernels',
    # # 'CORnet-S_train_gabor_scrumble' : 'Scrumble fit gabor kernels',
    # 'CORnet-S_train_gabor_multi_dist': 'Mixture gaussian gabor kernels',

    # Just rephrase layer 2:
    # 'CORnet-S_train_V2': 'Random init',
    # # 'CORnet-S_train_gabor_dist_second_corr_no_resize': 'Correlate Conv1',
    # # 'CORnet-S_train_gabor_dist_second_kernel_conv': 'Kernel-Convolution Conv1',
    # 'CORnet-S_train_gabor_dist_both_kernel': 'MG gabor kernels',
    # # 'CORnet-S_train_gabor_dist_weight_dist_kernel': "MG weight kernels",
    # 'CORnet-S_train_gabor_dist_weight_dist_channel': 'MG weight channels',
    # Round 5
    # 'CORnet-S_train_gabor_dist_both_kernel': 'MG gabor V1.Conv1, MG gabor kernel V1.conv2',
    # 'CORnet-S_train_gabor_dist_weight_dist_kernel': "MG gabor V1.Conv1, MG weights kernel V1.conv2, V2 -> train",
    # 'CORnet-S_train_gabor_dist_weight_dist_channel': 'MG gabor V1.Conv1, MG weights channel V1.conv2, V2 -> train',
    # # 'CORnet-S_train_IT_norm_dist' : 'Norm dist init, IT -> train'
    # 'CORnet-S_train_gabor_dist_kernel_gabor_dist_channel': 'MG gabor V1.Conv1, MG gabor channel V1.conv2',

    # # actually: gmk1_wmc2_wmk_3_wmk4_wrong
    # 'CORnet-S_train_gmk1_wmc2_wmk_3_wmk4_wrong': '',
    # Layer 3
    # Fixed: Layer 1: MG gabor, Layer 2: MG weight channels
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    #
    # 'CORnet-S_train_gmk1_wmc2_wmk_3': 'MG weight kernels',
    # 'CORnet-S_train_gmk1_wmc2_kn3':  'Kernel norm dist l3',
    # 'CORnet-S_train_gmk1_wmc2_ln3':  'Layer norm dist',
    # Layer 4
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    # 'CORnet-S_train_gabor_dist_kernel_weight_dist_channel_second_forth': 'MG weight kernels',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4' : 'Kernel norm dist l4',
    # 'CORnet-S_train_gmk1_wmc2_ln3_ln4': 'Layer norm dist',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4': 'Brain focus, kernel norm dist',
    # V1 best pat:
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    # Layer 5 :
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5':'Brain focus, kernel norm dist',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_ln5':'Brain focus, layer norm dist',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5': 'Kernel norm dist l5',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5' : 'Layer norm dist',
    # Layer 6 - brain focus
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6' : 'Brain focus, MG weight channels 1',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_ln5_wm6' : 'Brain focus, MG weight channels 2',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_ln5_ln6':'Brain focus, Layer norm dist',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_ln5_kn6': 'Brain focus, Kernel norm dist',
    # imagenet focus, but als brain focus:
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6' : 'MG weight channels 1', # different, this is one
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_wm6' : 'MG weight channels 2', # --> same gpu setting
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_ln6' : 'Layer norm dist', # different setting, this is one
    # 'CORnet-S_train_gmk1_cmc2_kn3_kn4_ln5_kn6' : 'Kernel norm dist', # different, this is one # imagenet focus, but als brain focus:
    # Actual imagenet focus
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wmc6_v2' : 'MG weight channels 1',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_wm6_v2' : 'Imagenet focus',#'MG weight channels 2',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_ln6_v2' : 'Layer norm dist',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_kn6_v3' : 'Kernel norm dist',

    # Layer 7
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wmc6_kn7_v2': 'Imagenet focus',  # 'Kernel norm dist 2',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_kn7': 'Brain focus, kernel norm dist',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_ln7': 'Brain benchmark focus',  # 'Brain focus, layer norm dist',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_wmc6_kn7' : 'Kernel norm dist 1',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_wmc6_ln7' : 'Layer norm dist 1',
    # 'CORnet-S_train_gmk1_wmc2_ln3_ln4_ln5_ln6_ln7_v2': 'Layer norm dist',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_kn6_kn7_v2' : 'Kernel norm dist',

    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wmc6_ln7' : 'Layer norm dist 2',
    # 'CORnet-S_train_gmk1_wmc2_wmk3_wmk4_wmk5_wmc6_wmk7_v2' : 'MG kernel + MG channel conv2',
    # 'CORnet-S_train_gmk1_wmc2_ln3_ln4_ln5_wm6_ln7': 'Layer norm dist + MG channel conv2',
    # GPU analysis:
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6': '2 GPUs',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_2_gpu': '2 GPUs again',  # different, this is one
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6': '1 GPU',  # different, this is one
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_1_gpu': 'Brain benchmark focus',  # different, this is one

    # Compare layer 7 plus train with base layer 7 and layer 6 trains
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_wmc6_ln7_v2' : 'Layer norm train 7',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_wmc6_kn7_v2' : "Kernel norm, train 7",
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wmc6_kn7_v2': 'Imagenet focus L7',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_wm6_v2' : 'Imagenet focus L6',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_ln7_v2' : 'Layer norm brain focus, train 7',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_ln7': 'Brain benchmark focus L7',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_1_gpu': 'Brain benchmark focus L6',
    # New layer 7 alternatives
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wmc6_kn7_v2': 'Norm dist Imagenet',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_ln7': 'Norm dist Brain',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_jc7' : 'Jumble channel Brain',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6_jc7':'Jumble channel Imagenet',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_jk7': 'Jumble kernel Brain',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6_jk7' : 'Jumble kernel Imagenet',
    # Distributions
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_ln7': 'Norm dist Brain',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_bdk7': 'Best distribution kernel',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_bdl7': 'Best distribution layer',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6_bdl7': 'Best distribution',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6_bdk7': 'Imagenet best distribution kernel',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wmc6_kn7_v2': 'Normal distribution',
    # Distirbutions various fixed:
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6_d1l7': 'Beta distribution',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6_d2l7': 'Pareto layer',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_d1l7' : 'Beta brain layer',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_d2l7' : 'Pareto brain layer',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6_d3l7': 'Gamma layer',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6_d5l7' : 'Uniform layer',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6_d6l7': 'Poisson layer',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6_d7l7' : 'Exponential layer',
    # Kernel
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6_d1k7' : 'Beta kernel',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6_d2k7' : 'Pareto kernel',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_d1k7' : 'Beta brain kernel',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_d2k7' : 'Pareto brain kernel',
    # # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6_d3k7' : 'Gamma kernel',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6_d5k7' : 'Uniform kernel',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6_d6k7' : 'Poisson kernel',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wm6_d7k7' : 'Exponential kernel',

    # V4
    # 'CORnet-S_train_IT_seed_0': 'Random IT -> train',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wmc6_kn7_v2': 'Best until V2',

    # 'CORnet-S_brain_kn8_kn9_kn10_wmk11_kn12' : 'Brain focus 2',
    # 'CORnet-S_train_kn8_kn9_kn10_wmc11_kn12' : 'Imagenet focus 1',
    # 'CORnet-S_train_kn8_kn9_kn10_wmk11_kn12': 'Train V2.conv3 Imagenet',
    # 'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12': 'Train V2.conv3, brain',
    # 'CORnet-S_train_kn8_kn9_kn10_wmk11': 'No in between train Imagenet',
    # 'CORnet-S_brain_kn8_kn9_kn10_wmc11': 'No in between train brain',
    # Batchnrom corrected

    #  Layer 1 & 2
    # 'CORnet-S_train_V2': 'Random init',
    # 'CORnet-S_train_gabor_dist_weight_dist_channel' : 'V1 base',
    # 'CORnet-S_train_gabor_dist_weight_dist_channel_ra_BF' : 'V1 no batchnorm',
    # 'CORnet-S_train_gabor_dist_weight_dist_channel_bi_BF' : 'V1 batchnorm',
    # 'CORnet-S_train_gabor_dist_both_kernel' : "V1 gabor ",
    # 'CORnet-S_train_gabor_dist_both_kernel_ra_BF' : "V1 gabor no batchnorm",
    # 'CORnet-S_train_gabor_dist_both_kernel_bi_BF' : "V1 gabor batchnorm",

    # Layer 3
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    # 'CORnet-S_train_gmk1_wmc2_ln3' : "V2.conv1 Layer norm dist",
    # 'CORnet-S_train_gmk1_wmc2_ln3_bi_BF' : "V2.conv1 Layer bn",
    # 'CORnet-S_train_gmk1_wmc2_ln3_ra_BF' : "V2.conv1 Layer no bn",
    # 'CORnet-S_train_gmk1_wmc2_kn3' : "V2.conv1 Kernel norm dist",
    # 'CORnet-S_train_gmk1_wmc2_kn3_bi_BF' : "V2.conv1 Kernel bn",
    # 'CORnet-S_train_gmk1_wmc2_kn3_ra_BF' : "V2.conv1 Kernel no bn",
    #
    # 'CORnet-S_train_gmk1_gmk2_ln3_bi' : 'V2.conv2 layer dist batchnorm',
    # 'CORnet-S_train_gmk1_gmk2_ln3_ra' : 'V2.conv2 layer dist no batchnorm',
    # 'CORnet-S_train_gmk1_gmk2_kn3_bi' : 'V2.conv2 kernel norm batchnorm',
    # 'CORnet-S_train_gmk1_gmk2_kn3_ra' : 'V2.conv2 kernel norm no batchnorm',

    # Layer 4
    # 'CORnet-S_train_V4': 'Random init(whole V2)',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_bi' : 'V2.skip Kernel norm dist bn',
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ra' : 'V2.skip Kernel norm dist no bn',
    #  'CORnet-S_train_gmk1_wmc2_kn3_ln4_bi' : 'V2.skip Layer norm dist bn',
    # 'CORnet-S_train_gmk1_wmc2_kn3_ln4_ra' : 'V2.skip Layer norm dist no bn',

    # Layer 5
    'CORnet-S_train_V4': 'Random init(whole V2)',
    'CORnet-S_train_gmk1_gmk2_ln3_ln4_ln5_ra': 'V2.conv1 layer norm dist',
    'CORnet-S_train_gmk1_gmk2_ln3_ln4_ln5_bi': 'V2.conv1 layer norm dist batchnorm',
    'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_ra': 'V2.conv1 kernel norm dist',
    'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_bi': 'V2.conv1 kernel norm dist batchnorm',
    # # Layer 6
    'CORnet-S_train_gmk1_gmk2_kn3_kn4_ln5_wm6_ra': 'V2.conv2 weight dist',
    'CORnet-S_train_gmk1_gmk2_kn3_kn4_ln5_wm6_bi': 'V2.conv2 weight dist batchnorm',

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
    # plot_first_epochs(best_models_brain, brain=False)

    # plot_single_benchmarks(best_models_imagenet, brain=True)
    # , 'CORnet-S_train_gabor_dist_both_kernel': 'V1.conv2'
    # plot_single_benchmarks({'CORnet-S_train_gabor_dist_weight_dist_channel': 'V1'}, compare_batchfix=True)
    # plot_first_epochs({'CORnet-S_train_gabor_dist_both_kernel_ra_BF': 'V1',
    #                         'CORnet-S_train_gabor_dist_both_kernel_bi_BF': 'V1 batchnorm',
    #                         # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_wm6_v2_BF' : 'V2.conv2',
    #                         'CORnet-S_full' : 'Full'
    #                         })
    # plot_single_benchmarks(['CORnet-S_full', 'CORnet-S_train_gabor_dist_both_kernel'], brain=False, compare_batchfix=True)
    # plot_single_benchmarks(best_models_brain, brain=False)
