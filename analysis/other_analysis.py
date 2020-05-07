import numpy as np

from benchmark.database import get_connection, load_scores
from nets.global_data import convergence_epoch, benchmarks
from plot.plot_data import plot_data_double, plot_data_base

models = {
    'resnet_v1_CORnet-S_brain_t7_t12_wmc15_IT_bi': 'layer4.2.conv3_special',
    'resnet_v1_CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wmc6_kn7_tr_bi': 'layer1.2.conv3_special'
}

random = {
    'resnet_v1_CORnet-S_full': 'Standard training',
    'resnet_v1_CORnet-S_train_random': 'Decoder',
}
random_alexnet = {
    'alexnet_v1_CORnet-S_full': 'Standard training',
    'alexnet_v1_CORnet-S_train_random': 'Decoder',
}

models_alexnet = {
    'alexnet_v1_CORnet-S_brain_t7_t12_wmc15_IT_bi': 'features.12',
    # 'resnet_v1_CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wmc6_kn7_tr_bi' : 'layer2.3.conv3_special'
}

layers_number = {
    'Standard training': 50,
    'layer4.2.conv3': 1,
    'layer4.2.conv3_special': 17,
    'layer1.2.conv3_special': 40,
    'Decoder': 1,
}

layers_number_alexnet = {
    'Standard training': 8,
    'features.12': 3,
    # 'layer4.2.conv3_special': 17,
    # 'layer1.2.conv3_special' : 40,
    'Decoder': 3,
}


def score_over_layers_avg(models, random, imagenet=False, convergence=False, model_name='resnet',
                          layers_number=layers_number):
    conn = get_connection()
    names = []

    for model in models.keys():
        if convergence and model in convergence_epoch:
            postfix = f'_epoch_{convergence_epoch[model]:02d}'
        else:
            postfix = f'_epoch_06'
        names.append(f'{model}{postfix}')

    # model_dict = load_scores(conn, names, benchmarks)
    data = {}
    layers = []

    full = 0
    for model in random.keys():
        if convergence:
            names.append(f'{model}_epoch_{convergence_epoch[model]:02d}')
        else:
            names.append(f'{model}_epoch_06')

    model_dict = load_scores(conn, names, benchmarks)
    data2 = {}
    data2['Score'] = []
    layers2 = []
    for model, layer in random.items():
        if convergence and model in convergence_epoch:
            postfix = f'_epoch_{convergence_epoch[model]:02d}'
        else:
            postfix = f'_epoch_06'
        layers2.append(layers_number[layer])
        if imagenet:
            if model == f'{model_name}_v1_CORnet-S_full':
                full = np.mean(model_dict[f'{model}{postfix}'][5])
                data2['Score'].append(100)
            else:
                percent = (np.mean(model_dict[f'{model}{postfix}'][5]) / full) * 100
                data2['Score'].append(percent)
        else:
            if model == f'{model_name}_v1_CORnet-S_full':
                full = np.mean(model_dict[f'{model}{postfix}'][2:5])
                data2['Score'].append(100)
            else:
                percent = (np.mean(model_dict[f'{model}{postfix}'][2:5]) / full) * 100
                data2['Score'].append(percent)

    data['Score'] = []
    for model, layer in models.items():
        if convergence and model in convergence_epoch:
            postfix = f'_epoch_{convergence_epoch[model]:02d}'
        else:
            postfix = f'_epoch_06'
        layers.append(layers_number[layer])
        if imagenet:
            if model == f'{model_name}_v1_CORnet-S_full':
                full = np.mean(model_dict[f'{model}{postfix}'][5])
                data['Score'].append(100)
            else:
                percent = (np.mean(model_dict[f'{model}{postfix}'][5]) / full) * 100
                data['Score'].append(percent)
        else:
            if model == f'{model_name}_v1_CORnet-S_full':
                full = np.mean(model_dict[f'{model}{postfix}'][2:5])
                data['Score'].append(100)
            else:
                percent = (np.mean(model_dict[f'{model}{postfix}'][2:5]) / full) * 100
                data['Score'].append(percent)

    if imagenet:
        title = f'{model_name} Imagenet score over layers'
    else:
        title = f'{model_name} Brain-Score Benchmark mean(V4, IT, Behavior) over layers'
    labels = [value.replace('special', 'trained') if 'special' in value else value for value in
              models.values()]
    plot_data_double(data, data2, title, x_name='Number of trained layers',
                     y_name='mean(V4, IT, Behavior)[% of standard training]', x_ticks=layers,
                     x_ticks_2=layers2, percent=True, data_labels=labels)


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
            else:
                frac = (np.mean(model_dict[f'{model}_epoch_{convergence_epoch[model]:02d}'][5]) / full) * 100
                scores.append(frac)
            x_values[name] = epochs + [convergence_epoch[model]]
        else:
            x_values[name] = epochs
        data[name] = scores

    title = f'{model_name} Brain scores mean vs epochs' if brain else f'{model_name} Imagenet score vs epochs'
    plot_data_base(data, title, x_values, 'Epochs', 'Score', x_ticks=epochs + [10, 20, 30],
                   percent=True, special_xaxis=True)


if __name__ == '__main__':
    plot_first_epochs({**models, **random}, [0, 6, 10, 20], False, convergence=True)
    plot_first_epochs({**models, **random}, [0, 6, 10, 20], True, convergence=True)
    score_over_layers_avg(models, random, False, convergence=True)
    score_over_layers_avg(models, random, True, convergence=True)
    plot_first_epochs({**models_alexnet, **random_alexnet}, [0, 6, 10, 20], False, convergence=True,
                      model_name='alexnet')
    plot_first_epochs({**models_alexnet, **random_alexnet}, [0, 6, 10, 20], True, convergence=True,
                      model_name='alexnet')
    score_over_layers_avg(models_alexnet, random_alexnet, False, convergence=True, model_name='alexnet',
                          layers_number=layers_number_alexnet)
    score_over_layers_avg(models_alexnet, random_alexnet, True, convergence=True, model_name='alexnet',
                          layers_number=layers_number_alexnet)
