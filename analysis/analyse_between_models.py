from benchmark.database import load_data
from plot.plot_data import plot_data, benchmarks, get_all_perturbations, get_list_all_models, \
    benchmarks_small


def precomputed():
    data = {'alexnet_random': [0.23957123, 0.22241723, 0.43034628, 0.26072557, 0.10744794, 0.0012],
            'alexnet_norm_dist': [0.21456451, 0.2003876, 0.44366336, 0.24848454, 0.10700565,
                                  0.0011],
            'alexnet': [0.2154463, 0.30824973, 0.54964844, 0.50798172, 0.37017073, 0.51908]}

    data = {'CORnet-S_jumbler': [0.14669811, 0.10989715, 0.40339382, 0.18461161, 0.01440615, 0.001],
            'CORnet-S_norm_dist': [0.17120103, 0.11540728, 0.4419885, 0.18780089, 0.06323756,
                                   0.00074],
            'CORnet-S_uniform_dist': [0.15909725, 0.10662348, 0.38189398, 0.15885385, 0.02673891,
                                      0.00106],
            'CORnet-S_random': [0.1310817, 0.08967254, 0.41193349, 0.18174943, -0.0087891, 0.001],
            'CORnet-S': [0.14632189, 0.20851341, 0.59320619, 0.53397797, 0.54545987, 0.74744]}

    benchmarks = ['movshon.FreemanZiemba2013.V1-pls', 'movshon.FreemanZiemba2013.V2-pls',
                  'dicarlo.Majaj2015.V4-pls',
                  'dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n',
                  'fei-fei.Deng2009-top1']

    plot_data(benchmarks, data, ['V1', 'V2', 'V4', 'IT', 'Behaviour', 'Imagenet'])


def model_perturbation_scores():
    # Per model results
    # models = ['CORnet-S', 'CORnet-S_random', 'CORnet-S_trained_incremental_init', 'CORnet-S_incremental_init']
    # models = get_list_all_pert(['CORnet-S'])
    # models = ['CORnet-S', 'CORnet-S_random_2', 'CORnet-S_random', 'CORnet-S_train_IT_random']
    models = ['CORnet-S', 'CORnet-S_train_all_epoch_20']
    # models = ['CORnet-S', 'CORnet-S_random']
    data = load_data(models, ['movshon.FreemanZiemba2013.V1-pls',
                              'movshon.FreemanZiemba2013.V2-pls'] + benchmarks)
    plot_data(['movshon.FreemanZiemba2013.V1-pls', 'movshon.FreemanZiemba2013.V2-pls'] + benchmarks,
              data,
              ['V1', 'V2', 'V4', 'IT', 'Behaviour', 'Imagenet'], 'cornet_IT_train_all')


def small_model_perturbation_score():
    # Per model results
    # models = ['CORnet-S', 'CORnet-S_random', 'CORnet-S_norm_dist', 'CORnet-S_std_function_1', 'CORnet-S_std_function_2',
    #           'CORnet-S_std_function_3', 'CORnet-S_std_function_4']
    # models = ['CORnet-S', 'CORnet-S_random', 'CORnet-S_trained_incremental_init', 'CORnet-S_incremental_init']
    # models = ['CORnet-S', 'CORnet-S_random', 'CORnet-S_train_norm_dist', 'CORnet-S_train_norm_dist_kernel']
    # models = ['CORnet-S', 'CORnet-S_random_2', 'CORnet-S_train_IT_random']
    models = ['CORnet-S', 'CORnet-S_train_all_epoch_20']
    # models = get_list_all_pert(['CORnet-S'])
    data = load_data(models,
                     (['dicarlo.Majaj2015.V4-pls'] + benchmarks_small + ['fei-fei.Deng2009-top1']))
    # plot_data(benchmarks, data, ['V4', 'IT', 'Behaviour', 'Imagenet'], 'alexnet_all')
    plot_data(['dicarlo.Majaj2015.V4-pls'] + benchmarks_small + ['fei-fei.Deng2009-top1'], data,
              ['V4', 'IT', 'Behaviour', 'Imagenet'], 'delete')


def correlation_per_all_perturbation():
    # Model correlation per perturbation: random
    for v in get_all_perturbations():
        correlate_per_perturbation(v)


def correlate_per_perturbation(perturbation):
    models = get_list_all_models([perturbation])
    data = load_data(models, benchmarks)
    plot_data(benchmarks, data, ['V4', 'IT', 'Behaviour', 'Imagenet'], f'corr{perturbation}')


if __name__ == '__main__':
    # precomputed_smaller_benchmarkset()
    # correlation_per_all_perturbation()
    small_model_perturbation_score()
    # model_perturbation_scores()
