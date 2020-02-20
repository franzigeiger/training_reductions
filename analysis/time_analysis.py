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
                              'dicarlo.Rajalingham2018-i2n',
                              'fei-fei.Deng2009-top1'])
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


if __name__ == '__main__':
    plot_benchmarks_over_epochs('resnet_mil_trained', (0, 1, 2, 3, 4, 5, 10, 15, 20, 90))
    # plot_benchmarks_over_epochs('CORnet-S_train_all')
    # plot_benchmarks_over_epochs('CORnet-S_full', (0, 1, 2, 3, 5, 7, 10, 15, 20, 43))
    # plot_benchmarks_over_epochs('CORnet-S_train_IT_seed_0', (5, 10, 15, 20))
    # plot_benchmarks_over_epochs('CORnet-S_train_gabor', (1, 3, 5, 10, 15))
    # plot_benchmarks_over_epochs('CORnet-S_train_gabor_reshape', (1, 3, 5, 10, 15))
    # plot_benchmarks_over_epochs('CORnet-S_train_second', (1, 2, 3, 5, 7))
    # plot_benchmarks_over_epochs('CORnet-S_train_second_no_batchnorm', (1, 2, 3, 5, 10, 15))
    # plot_over_epochs(['CORnet-S_train_all', 'CORnet-S_full'])
    # plot_over_epochs(['CORnet-S_train_all', 'CORnet-S_train_IT_seed_0', 'CORnet-S_gabor', 'CORnet-S_gabor_reshape'])
    # models = {'CORnet-S_full' : 'Train all',
    #           # 'CORnet-S_train_V2': 'Random init, V1.conv2-> train',
    #           # 'CORnet-S_train_gabor': 'Fix gabors, V1.conv2 -> train',
    #           # 'CORnet-S_train_gabor_reshape':'Fix gabors+normalize, V1.conv2 -> train',
    #           # 'CORnet-S_train_second': 'Fix gabors and Correlate, V2 -> train',
    #           'CORnet-S_train_IT_seed_0':'Random init, IT -> train',
    #           'CORnet-S_rand_conv1':'Random init V1.Conv1, V1.Conv2-> train',
    #           'CORnet-S_train_second_corr_only':'Trained Conv1, Correlate Conv2, V2 -> train',
    #           'CORnet-S_train_gabor_fit':'Fit gabor Conv1, V1.Conv2 -> train',
    #           'CORnet-S_train_gabor_fit_second_corr':'Fit gabor Conv1, Correlate Conv2, V2 -> train'
    #
    #           # 'CORnet-S_train_second_no_batchnorm': 'Fix gabors and Correlate, Correlate, V2 -> train'
    #           # 'CORnet-S_train_second':''
    #           }
    # plot_models_benchmarks(models, 'first_generation')
