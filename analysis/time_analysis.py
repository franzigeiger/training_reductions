from benchmark.database import load_scores
from plot.plot_data import get_connection, plot_data_base


def plot_over_epochs(models):
    model_dict = {}
    conn = get_connection()
    epochs = (5, 10)
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
    model_dict['CORnet-S'] = load_scores(conn, ['CORnet-S'],
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
        data['CORnet-S'] = [0] * 3 + [model_dict['CORnet-S']['CORnet-S'][i]]
        plot_data_base(data, f'{benchmarks[i]} Benchmark over epochs', epochs, 'Score over epochs', 'Score')


def plot_benchmarks_over_epochs(model, epochs=None):
    model_dict = {}
    conn = get_connection()
    if epochs is None:
        epochs = (5, 10, 15, 20)

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
    model_dict['CORnet-S'] = load_scores(conn, ['CORnet-S'],
                                         ['movshon.FreemanZiemba2013.V1-pls',
                                          'movshon.FreemanZiemba2013.V2-pls',
                                          'dicarlo.Majaj2015.V4-pls',
                                          'dicarlo.Majaj2015.IT-pls',
                                          'dicarlo.Rajalingham2018-i2n',
                                          'fei-fei.Deng2009-top1'])
    benchmarks = ['V1', 'V2', 'V4', 'IT', 'Behavior', 'Imagenet']
    data = {bench: [] for bench in benchmarks}
    for i in range(6):
        for epoch in epochs:
            data[benchmarks[i]].append(model_dict[model][f'{model}_epoch_{epoch:02d}'][i])
        # data['CORnet-S'] = [0]*3 + [model_dict['CORnet-S']['CORnet-S'][i]]
    # plot_data_base(data, f'Resnet50(Michael version) Brain-Score benchmarks', epochs, 'Epoch', 'Score',x_ticks=epochs, log=True)
    # plot_data_base(data, f'CORnet-S(Franzi version) Brain-Score benchmarks', epochs, 'Epoch', 'Score', x_ticks=epochs)
    plot_data_base(data, f'{model} Brain-Score benchmarks', epochs, 'Epoch', 'Score', x_ticks=epochs)


if __name__ == '__main__':
    # plot_benchmarks_over_epochs('resnet_mil_trained', (1, 5, 10, 15, 20, 90))
    # plot_benchmarks_over_epochs('CORnet-S_train_all')
    plot_benchmarks_over_epochs('CORnet-S_train_IT_seed_0', (5, 10, 15, 20))
    plot_benchmarks_over_epochs('CORnet-S_train_gabor', (1, 3, 5, 10, 15))
    plot_benchmarks_over_epochs('CORnet-S_train_gabor_reshape', (1, 3, 5, 10, 15))
    plot_benchmarks_over_epochs('CORnet-S_train_second', (1, 2, 3, 5, 7))
    plot_benchmarks_over_epochs('CORnet-S_train_second_no_batchnorm', (1, 2, 3, 5, 10, 15))
    # plot_over_epochs(['CORnet-S_train_all', 'CORnet-S_train_IT_seed_0', 'CORnet-S_gabor', 'CORnet-S_gabor_reshape'])
