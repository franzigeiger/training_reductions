from torch import nn

from nets import get_config, apply_generic_other, conv_to_norm, global_data
from nets.test_models import get_resnet50, run_model_training, get_alexnet
from utils.models import mapping_1, mapping_2, alexnet_mapping

to_train = []


def train_other(template='CORnet-S_brain2_t7_t12_knall_IT_bi', net='resnet', version='v1', train_func=None):
    config = get_config(template)
    if net == 'resnet':
        model = get_resnet50(False)
        if version == 'v1':
            mapping = mapping_1
        elif version == 'v2':
            mapping = mapping_2
        else:
            mapping = mapping_1
            del config['bn_init']
            del config['batchnorm']
    if net == 'vgg':
        # model = get_resnet50(False)
        mapping = None  # to be done
    if net == 'alexnet':
        model = get_alexnet(False)

        mapping = alexnet_mapping

    other_config = create_config(mapping, config, model)
    identifier = f'{net}_{version}_{template}'
    if global_data.seed != 0:
        identifier = f'{identifier}_seed{global_data.seed}'
    model = apply_generic_other(model, other_config)
    run_model_training(model=model, identifier=identifier, config=other_config, train_func=train_func)


def create_config(mapping, config, model):
    resnet_config = config
    to_train = []
    for layer, m in model.named_modules():
        # if not any(module in layer for module in config['layers']):
        if type(m) == nn.BatchNorm2d or type(m) == nn.Conv2d or type(m) == nn.Linear:
            if layer not in mapping:
                # print(f'Couldn\'t find specification for layer {layer}')
                print(layer)
                to_train.append(layer)
            else:
                cornet = mapping[layer]
                if cornet in config:
                    resnet_config[layer] = cornet
                    # resnet_config[f'{layer}_func'] = config[cornet]
                if any(layer in cornet for layer in config['layers']):
                    to_train.append(layer)
                elif type(m) == nn.BatchNorm2d:
                    base = conv_to_norm[cornet]
                    if base in config:
                        resnet_config[layer] = cornet
                    if any(layer in base for layer in config['layers']):
                        to_train.append(layer)
                else:
                    print(layer)
        else:
            print(layer)
    resnet_config['layers'] = to_train
    return resnet_config
