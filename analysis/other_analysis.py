from itertools import chain

import numpy as np

from base_models.global_data import convergence_epoch, benchmarks
from benchmark.database import get_connection, load_scores, load_error_bared
from plot.plot_data import plot_data_double, plot_data_base, green_palette

models = {
    # 'resnet_v1_CORnet-S_brain_t7_t12_wmc15_IT_bi': 'layer4.2.conv3_special',
    'resnet_v3_CORnet-S_cluster2_v2_IT_trconv3_bi': 'layer4.2.conv3_special',
    # 'resnet_v3_CORnet-S_cluster2_v2_V4_trconv3_bi': 'layer3.5.conv3_special',
    # 'resnet_v1_CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_tr_bi' : 'layer3.5.bn3_special',
    # 'resnet_v3_CORnet-S_train_gmk1_cl2_7_7tr_bi': 'layer1.2.conv3_special'
}

models_mobilenet = {
    'mobilenet_v1_CORnet-S_cluster2_v2_IT_trconv3_bi': 'V2.special',
    'mobilenet_v1_CORnet-S_cluster2_v2_V4_trconv3_bi': 'V4.special',
    'mobilenet_v1_CORnet-S_train_gmk1_cl2_7_7tr_bi': 'IT.special'
}

random = {
    'resnet_v1_CORnet-S_full': 'Standard training',
    'resnet_v1_CORnet-S_train_random': 'Decoder',
}

random_mobilenet = {
    'mobilenet_v1_1.0_224': 'Standard training',
    'mobilenet_random': 'Decoder',
}
random_alexnet = {
    'alexnet_v1_CORnet-S_full': 'Standard training',
    'alexnet_v1_CORnet-S_train_random': 'Decoder',
}

models_alexnet = {
    # 'alexnet_v1_CORnet-S_brain_t7_t12_wmc15_IT_bi': 'features.12',
    'alexnet_v4_CORnet-S_cluster2_v2_IT_trconv3_bi': 'features.12',
    # 'alexnet_v1_CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_tr_bi': 'features.8',
    'alexnet_v4_CORnet-S_cluster2_v2_V4_trconv3_bi': 'features.8',
    'alexnet_v4_CORnet-S_train_gmk1_cl2_7_7tr_bi': 'features.6'
    # 'alexnet_v1_CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wmc6_kn7_tr_bi': 'features.6'
}

layers_number = {
    'Standard training': 50,
    'layer4.2.conv3': 1,
    'layer4.2.conv3_special': 17,
    'layer3.5.conv3_special': 21,
    'layer1.2.conv3_special': 40,
    'Decoder': 1,
}

layers_number_mobilenet = {
    'Standard training': 28,
    'V2.special': 21,
    'V4.special': 15,
    'IT.special': 12,
    'Decoder': 1,
}

layers_number_alexnet = {
    'Standard training': 8,
    'features.12': 3,
    'features.8': 4,
    'features.6': 5,
    'Decoder': 3,
}

all_layers = {
    'Alexnet': 8,
    'Resnet50': 50
}

label = 'Artificial genome + critical training(AG + CT)'


def score_over_layers_avg(models_resnet, random, models_alexnet={}, random_alexnet={}, imagenet=False,
                          convergence=False,
                          model_name='resnet',
                          layers_numbers=[layers_number, layers_number, layers_number_mobilenet,
                                          layers_number_mobilenet],
                          gs=None, ax=None, selection=[]):
    conn = get_connection()
    full = 0
    model_dict = load_error_bared(conn, list(
        chain(models_resnet.keys(), models_mobilenet.keys(), random.keys(), random_mobilenet.keys())), benchmarks,
                                  convergence=convergence)
    data = {}
    err = {}
    layers = {}
    labels = {}
    idx = 0
    for models, label in zip([random, models_resnet],  # , random_mobilenet, models_mobilenet
                             ['Resnet50 KN+DT', 'Resnet50 Transfer AG+CT', 'Alexnet KN+DT', 'Alexnet Transfer AG+CT']):
        data[label] = []
        layers[label] = []
        layers_number = layers_numbers[idx]
        idx += 1
        for model, layer in models.items():
            layers[label].append(layers_number[layer])
            if model == f'{model_name}_v1_CORnet-S_full':
                full = np.mean(model_dict[model][selection])
                data[label].append(100)
            else:
                percent = (np.mean(model_dict[model][selection]) / full) * 100
                data[label].append(percent)
            full_err = (np.mean(model_dict[model][:6][selection]) / full) * 100
            err[label] = full_err
        if 'Alexnet' in label:
            labels[label] = models.values()
        else:
            labels[label] = [value.split('.')[0] for value in models.values()]

    if imagenet:
        title = f'{model_name} Imagenet score over layers'
        y = 'Imagenet [% of standard training]'
    else:
        title = f'{model_name} Brain-Score Benchmark mean(V4, IT, Behavior) over layers'
        if len(selection) == 3:
            y = r"mean(V4, IT, Behavior) [% of standard training]"
        else:
            y = r"mean(V1,V2,V4,IT,Behavior) [% of standard training]"
    plot_data_double(data, {}, '', x_name='Number of trained layers [% of all layers]', y_name=y, x_ticks=layers,
                     x_ticks_2=[], percent=False, percent_x=True,
                     pal=['#424949'] + [green_palette[1]] + ['#ABB2B9'] + [green_palette[0]], data_labels=labels, gs=gs,
                     ax=ax)

def plot_first_epochs(models, epochs=None, brain=True, convergence=True, model_name='resnet'):
    model_dict = {}
    conn = get_connection()
    if epochs is None:
        epochs = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6)
    data = {}
    x_values = {}
    conv_number = convergence_epoch[f'{model_name}_v1_CORnet-S_full']
    full_tr = load_scores(conn, [f'{model_name}_v1_CORnet-S_full_epoch_{conv_number}'], benchmarks)[
        f'{model_name}_v1_CORnet-S_full_epoch_{conv_number}']
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
                y = 'mean(V4, IT, Behavior) [% of standard training]'
            else:
                frac = (np.mean(model_dict[f'{model}_epoch_{convergence_epoch[model]:02d}'][5]) / full) * 100
                y = 'Imagenet [% of standard training]'
                scores.append(frac)
            x_values[name] = epochs + [convergence_epoch[model]]
        else:
            x_values[name] = epochs
        data[name] = scores

    title = f'{model_name} Brain scores mean vs epochs' if brain else f'{model_name} Imagenet score vs epochs'
    plot_data_base(data, '', x_values, 'Epochs', y, x_ticks=epochs + [10, 20, 30],
                   percent=True, special_xaxis=True, only_blue=False)

if __name__ == '__main__':
    plot_first_epochs({**models, **random}, [0, 6, 10, 20], False, convergence=True)
    plot_first_epochs({**models, **random}, [0, 6, 10, 20], True, convergence=True)
    score_over_layers_avg(models, random, False, convergence=True)
    score_over_layers_avg(models, random, True, convergence=True)
    # plot_first_epochs({**models_alexnet, **random_alexnet}, [0, 6, 10, 20], False, convergence=True,
    #                   model_name='alexnet')
    # plot_first_epochs({**models_alexnet, **random_alexnet}, [0, 6, 10, 20], True, convergence=True,
    #                   model_name='alexnet')
    # score_over_layers_avg(models_alexnet, random_alexnet, False, convergence=True, model_name='alexnet',
    #                       layers_number=layers_number_alexnet)
    # score_over_layers_avg(models_alexnet, random_alexnet, True, convergence=True, model_name='alexnet',
    #                       layers_number=layers_number_alexnet)
