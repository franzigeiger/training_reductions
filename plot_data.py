

import numpy as np
import matplotlib.pyplot as plt


def precomputed():
    data = {'alexnet_random': [0.23957123, 0.22241723, 0.43034628, 0.26072557, 0.10744794, 0.0012],
            'alexnet_norm_dist': [0.21456451,0.2003876, 0.44366336, 0.24848454, 0.10700565, 0.0011],
            'alexnet': [0.2154463,0.30824973,0.54964844,0.50798172,0.37017073,0.51908]}

    data = {'CORnet-S_jumbler': [0.14669811, 0.10989715,0.40339382,0.18461161,0.01440615,0.001],
            'CORnet-S_norm_dist': [0.17120103,0.11540728,0.4419885,0.18780089,0.06323756,0.00074],
            'CORnet-S_uniform_dist': [0.15909725,0.10662348,0.38189398,0.15885385,0.02673891,0.00106],
            'CORnet-S_random': [0.1310817,0.08967254,0.41193349,0.18174943,-0.0087891,0.001],
            'CORnet-S': [0.14632189,0.20851341,0.59320619,0.53397797,0.54545987,0.74744]}

    benchmarks = ['movshon.FreemanZiemba2013.V1-pls', 'movshon.FreemanZiemba2013.V2-pls','dicarlo.Majaj2015.V4-pls', 'dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n', 'fei-fei.Deng2009-top1']

    plot_data(benchmarks, data, ['V1', 'V2', 'V4', 'IT', 'Behaviour', 'Imagenet'])


def precomputed_smaller_benchmarkset():
    data = {
            # 'CORnet-S_jumbler_V1': [0.47733455,0.19090331,0.01513933,0.00108],
            'CORnet-S_jumbler' : [0.49369259,0.19139916,0.10985487,0.001],
            # 'CORnet-S_norm_dist_V1':[0.40980232,0.17133552,0.05271374,0.001],
            'CORnet-S_norm_dist':[0.41582354,0.15079721,0.01790043,0.001],
            # 'CORnet-S_uniform_dist_V1':[0.39693421,0.1569587,0.03124594,0.001],
            'CORnet-S_uniform_dist':[0.41563981,0.13187024,0.01476676,0.001],
            'CORnet-S_random': [0.4294747,0.18823061,0.03713337,0.001],
            'CORnet-S': [0.58120619,0.42997797,0.54512493,0.69244]}
    benchmarks = ['dicarlo.Majaj2015.V4-pls', 'dicarlo.Majaj2015.IT-pls', 'dicarlo.Rajalingham2018-i2n', 'fei-fei.Deng2009-top1']

    plot_data(benchmarks, data, ['V4', 'IT', 'Behaviour', 'Imagenet'])


def plot_data(benchmarks, data, labels):
    # res, ax = plt.subplots()
    y = np.array([0,1,2])
    x = np.arange(len(benchmarks))
    plt.xticks(x, labels,rotation='vertical', fontsize=8)
    # plt.yticks(y, models)
    # plt.setlabel(xlabel='Models', ylabel='Benchmarks')
    print(data)
    for key, value in data.items():
        plt.plot(x, value, label=key, linestyle="",marker="o")
    plt.legend()
    # res.save('test.png')
    plt.savefig('foo.png')
    plt.show()


if __name__ == '__main__':
    precomputed_smaller_benchmarkset()