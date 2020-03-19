from benchmark.database import load_scores
from plot.plot_data import get_connection, plot_data_base, plot_bar_benchmarks


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
    epoch = 5
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
    'CORnet-S_train_V4': 'Random init(whole V2)',
    'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wmc6_kn7_v2': 'Imagenet focus',  # 'Kernel norm dist 2',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_kn7': 'Brain focus, kernel norm dist',
    'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_ln7': 'Brain benchmark focus',  # 'Brain focus, layer norm dist',
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

    'CORnet-S_full': 'Train all',
    # epoch 43
    # 'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_wm6_full' : 'Imagenet optimized until V2.conv2',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_full' : 'Brain benchmark optimized until V2.conv2'



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
