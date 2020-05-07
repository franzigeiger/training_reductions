import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from analysis.time_analysis import benchmarks, image_scores
from benchmark.database import get_connection, load_scores, load_model_parameter
from nets.global_data import best_brain_avg, layer_random, best_special_brain, best_models_brain_avg_all, random_scores
from plot.plot_data import plot_data_double
from runtime.compression import get_params


def plot_performance(imagenet=True, entry_models=[best_brain_avg], ax=None):
    # conn = get_connection()
    conn = get_connection()
    names = []
    for model in random_scores.keys():
        if model != "CORnet-S_random":
            names.append(f'{model}_epoch_06')
        else:
            names.append(model)
    performance = load_model_parameter(conn)
    model_dict = load_scores(conn, names, benchmarks)
    # performance2 = load_model_parameter(conn)

    time2 = []
    data2 = {'Score': []}
    if imagenet:
        high = np.mean(model_dict[f'CORnet-S_full_epoch_06'][5])
        for model, layer in random_scores.items():
            if model == "CORnet-S_random":
                perc = (np.mean(model_dict[f'{model}'][5]) / high) * 100
                data2['Score'].append(perc)
                time2.append(0)
            else:
                perc = (np.mean(model_dict[f'{model}_epoch_06'][5]) / high) * 100
                if layer in performance:
                    data2['Score'].append(perc)
                    time2.append(performance[layer])
    else:
        high = np.mean(model_dict[f'CORnet-S_full_epoch_06'][2:5])
        for model, layer in random_scores.items():
            # if model != "CORnet-S_random":
            if model == "CORnet-S_random":
                perc = (np.mean(model_dict[f'{model}'][2:5]) / high) * 100
                data2['Score'].append(perc)
                time2.append(0)
            else:
                perc = (np.mean(model_dict[f'{model}_epoch_06'][2:5]) / high) * 100
                if layer in performance:
                    data2['Score'].append(perc)
                    time2.append(performance[layer])

    data = {}
    time = {}
    labels = {}
    for entry_model, name in zip(entry_models, ['Selective training', 'Consecutive Training']):
        names = []
        for model in entry_model.keys():
            if model != "CORnet-S_random":
                names.append(f'{model}_epoch_06')
            else:
                names.append(model)
        model_dict = load_scores(conn, names, benchmarks)
        time[name] = []
        data[name] = []
        if imagenet:
            # high = np.mean(model_dict[f'CORnet-S_full_epoch_06'][5])
            for model, layer in entry_model.items():
                perc = (np.mean(model_dict[f'{model}_epoch_06'][5]) / high) * 100
                if layer in performance:
                    data[name].append(perc)
                    time[name].append(performance[layer])
        else:
            # high = np.mean(model_dict[f'CORnet-S_full_epoch_06'][2:5])
            for model, layer in entry_model.items():
                # if model != "CORnet-S_random":
                perc = (np.mean(model_dict[f'{model}_epoch_06'][2:5]) / high) * 100
                if layer in performance:
                    data[name].append(perc)
                    time[name].append(performance[layer])
        label = [value.replace('special', 'trained') if 'special' in value else value for value in entry_model.values()]
        labels[name] = label

    if imagenet:
        title = f'Imagenet score vs training time'
        y = 'Imagenet performance [% of standard training]'
    else:
        title = f'Brain-Score Benchmark mean(V4, IT, Behavior) vs training time'
        y = 'mean(V4, IT, Behavior)[% of standard training]'
    plot_data_double(data, data2, title, x_name='Training time', x_labels=[], y_name=y, x_ticks=time,
                     x_ticks_2=time2, percent=True, data_labels=labels, ax=ax)


def plot_num_params(imagenet=False, entry_models=[], ax=None):
    conn = get_connection()
    names = []
    for model in layer_random.keys():
        if model != "CORnet-S_random":
            names.append(f'{model}_epoch_06')
        else:
            names.append(model)
    data = {'Score': []}
    model_dict = load_scores(conn, names, benchmarks)
    full = 0
    for model, layer in layer_random.items():
        if imagenet:
            if model == "CORnet-S_random":
                percent = (np.mean(model_dict[f'{model}'][5]) / full) * 100
                data['Score'].append(percent)
            elif model == 'CORnet-S_full':
                full = np.mean(model_dict[f'{model}_epoch_06'][5])
                data['Score'].append(100)
            else:
                percent = (np.mean(model_dict[f'{model}_epoch_06'][5]) / full) * 100
                data['Score'].append(percent)
        else:
            if model == "CORnet-S_random":
                percent = (np.mean(model_dict[f'{model}'][2:5]) / full) * 100
                data['Score'].append(percent)
            elif model == 'CORnet-S_full':
                full = np.mean(model_dict[f'{model}_epoch_06'][2:5])
                data['Score'].append(100)
            else:
                percent = (np.mean(model_dict[f'{model}_epoch_06'][2:5]) / full) * 100
                data['Score'].append(percent)
    data2 = {}
    labels = {}
    params = {}
    for entry_model, name in zip(entry_models, ['Selective training', 'Consecutive Training']):
        data2[name] = []
        params[name] = []
        for model in entry_model.keys():
            print(model)
            if model.endswith('BF'):
                model = model.replace('_BF', '')
            if model == "CORnet-S_random":
                params[name].append(0)
            else:
                params[name].append(get_params(model))

        names = []
        for model in entry_model.keys():
            if model != "CORnet-S_random":
                names.append(f'{model}_epoch_06')
            else:
                names.append(model)
        model_dict = load_scores(conn, names, benchmarks)

        for model, layer in entry_model.items():
            if imagenet:
                if model == "CORnet-S_random":
                    percent = (np.mean(model_dict[f'{model}'][5]) / full) * 100
                    data2[name].append(percent)
                else:
                    percent = (np.mean(model_dict[f'{model}_epoch_06'][5]) / full) * 100
                    data2[name].append(percent)
            else:
                if model == "CORnet-S_random":
                    percent = (np.mean(model_dict[f'{model}'][2:5]) / full) * 100
                    data2[name].append(percent)
                else:
                    percent = (np.mean(model_dict[f'{model}_epoch_06'][2:5]) / full) * 100
                    data2[name].append(percent)
        label = [value.replace('special', 'trained') if 'special' in value else value for value in entry_model.values()]
        labels[name] = label

    params2 = []
    for model in layer_random.keys():
        print(model)
        if model.endswith('BF'):
            model = model.replace('_BF', '')
        if model == "CORnet-S_random":
            params2.append(0)
        else:
            params2.append(get_params(model))
    if imagenet:
        title = f'Imagenet score vs number of parameter'
        y = 'Imagenet performance [% of standard training]'
    else:
        title = f'Brain-Score Benchmark mean(V4, IT, Behavior) vs number of parameter'
        y = 'mean(V4, IT, Behavior)[% of standard training]'

    plot_data_double(data2, data, title, x_name='Number of parameters', x_labels=[], y_name=y, x_ticks=params,
                     x_ticks_2=params2, percent=True, data_labels=labels, ax=ax)

    # scatter_plot(params,data, x_label='Number of parameters', y_label='Score', #labels=entry_model.keys(),
    #              title='Training time vs scores', percent=True)


def plot_figure_3():
    sns.set_style("whitegrid", {'grid.color': '.95', })
    sns.set_context("talk")
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    image_scores('CORnet-S_brain_t7_t12_wmc15_IT_bi', 'CORnet-S_full', [100, 1000, 10000, 100000, 500000], brain=True,
                 ax=ax3)
    plot_num_params(imagenet=False, entry_models=[best_special_brain, best_models_brain_avg_all], ax=ax1)
    plot_performance(imagenet=False, entry_models=[best_special_brain, best_models_brain_avg_all], ax=ax2)
    plt.savefig(f'figure_3.png')
    plt.show()


if __name__ == '__main__':
    # plot_performance()
    # plot_performance(False)
    # plot_num_params()
    # plot_num_params(True)
    # plot_performance(entry_model=best_special_brain)
    # plot_performance(False, entry_model=best_special_brain)
    plot_figure_3()
    # plot_num_params(entry_model=best_special_brain)
    # plot_num_params(True, entry_model=best_special_brain)
