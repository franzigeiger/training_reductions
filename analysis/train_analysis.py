import os
import pickle

from plot.plot_data import plot_data_map, plot_date_map_custom_x

dir = '/braintree/home/fgeiger/weight_initialization/nets/model_weights'


def plot_training_effort(name):
    path = os.path.join(dir, f'results_{name}.pkl')
    file = open(path, 'rb')
    list = pickle.load(file)
    epochs = []
    epochs_train = []
    val_top1 = []
    val_top5 = []
    lr = []
    loss = []
    train_loss = []
    train_top1 = []
    for i in list:
        if 'val' in i:
            epochs.append(i['meta']['epoch'])
            val_top1.append(i['val']['top1'])
            val_top5.append(i['val']['top5'])
            loss.append(i['val']['loss'])
        if 'train' in i:
            epochs_train.append(i['meta']['epoch'])
            lr.append(i['train']['learning_rate'])
            train_loss.append(i['train']['loss'])
            train_top1.append(i['train']['top1'])
    plot_date_map_custom_x({'val_top1': val_top1, 'val_top5': val_top5, 'epoch': epochs},
                           f'validation {name}', label_field='epoch')
    plot_data_map({'learning_rate': lr, 'epoch': epochs_train, 'Top1': train_top1}, f'learning rate {name}',
                  label_field='epoch')
    print(f'Model {name} top1 at the end: {val_top1[-1]}')


def compare(name2, name1='CORnet-S_full', name3=None):
    epochs = []
    val_top1_1 = []
    val_top1_2 = []
    path = os.path.join(dir, f'results_{name2}.pkl')
    file = open(path, 'rb')
    list = pickle.load(file)
    for i in list:
        if 'val' in i and i['meta']['epoch'] not in epochs:
            epochs.append(i['meta']['epoch'])
            val_top1_2.append(i['val']['top1'])

    path = os.path.join(dir, f'results_{name1}.pkl')
    file = open(path, 'rb')
    list = pickle.load(file)
    for i in list:
        if 'val' in i:
            if i['meta']['epoch'] in epochs and len(val_top1_1) < len(val_top1_2):
                val_top1_1.append(i['val']['top1'])
    if name3:
        val_top1_3 = []
        path = os.path.join(dir, f'results_{name3}.pkl')
        file = open(path, 'rb')
        list = pickle.load(file)
        for i in list:
            if 'val' in i:
                if i['meta']['epoch'] in epochs and len(val_top1_3) < len(val_top1_2):
                    val_top1_3.append(i['val']['top1'])
        # val_top1_3.append(val_top1_3[-1])
        plot_date_map_custom_x({'Full train': val_top1_1, 'Imagenet optimized until V2.conv2': val_top1_2,
                                'Brain benchmark optimized until V2.conv2': val_top1_3, 'epoch': epochs},
                               f'Training behavior', label_field='epoch', y_name='Validation accuracy', x_name='Epoch')


if __name__ == '__main__':
    # plot_training_effort('CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wmc6_kn7_v2')
    # plot_training_effort('CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_full')
    # plot_training_effort('CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_wm6_full')
    # plot_training_effort('CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wmc6_full')
    # plot_training_effort('CORnet-S_full')
    # compare('CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wmc6_full')
    compare('CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_wm6_full', name3='CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_full')
    # compare('CORnet-S_train_gabor_multi_dist')
