from torch import nn

from base_models import get_config, apply_generic_other, conv_to_norm, global_data
from base_models.mobilenet import get_mobilenet
from base_models.test_models import get_resnet50, run_model_training, get_alexnet
from utils.models import mapping_1, mapping_2, alexnet_mapping, alexnet_mapping_2, mobilenet_mapping, \
    mobilenet_mapping_2, mobilenet_mapping_3, mobilenet_mapping_4, mobilenet_mapping_5, mobilenet_mapping_6

to_train = []


def train_other(template='CORnet-S_brain2_t7_t12_knall_IT_bi', net='resnet', version='v1', train_func=None):
    config = get_config(template)
    add_layers = []
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
        if version == 'v3':
            del config['bn_init']
            del config['batchnorm']
        if version == 'v4':
            config['no_bn'] = True
            del config['bn_init']
            del config['batchnorm']
        if version == 'v4':
            config['no_bn'] = True
            del config['bn_init']
            del config['batchnorm']
            mapping = alexnet_mapping_2

    if net == 'mobilenet':
        model = get_mobilenet(False)
        if version == 'v1':
            mapping = mobilenet_mapping
        elif version == 'v2':
            mapping = mobilenet_mapping_2
        elif version == 'v3':
            mapping = mobilenet_mapping_3
        elif version == 'v4':
            mapping = mobilenet_mapping_4
        elif version == 'v5':
            mapping = mobilenet_mapping_5
            add_layers = ['model.2.0', 'model.2.1', 'model.6.0', 'model.6.1', 'model.12.0', 'model.12.1', 'fc',
                          'decoder']
        elif version == 'v6':
            mapping = mobilenet_mapping_6
            add_layers = ['model.2.0', 'model.2.1', 'model.6.0', 'model.6.1', 'model.12.0', 'model.12.1', 'fc',
                          'decoder']
        elif version == 'v7':
            mapping = mobilenet_mapping_6
        config['no_bn'] = True
        del config['bn_init']
        del config['batchnorm']
    other_config = create_config(mapping, config, model)
    identifier = f'{net}_{version}_{template}'
    if global_data.seed != 0:
        identifier = f'{identifier}_seed{global_data.seed}'
    print(f'Run model {identifier}')
    model = apply_generic_other(model, other_config)
    other_config['layers'] = other_config['layers'] + add_layers
    run_model_training(model=model, identifier=identifier, config=other_config, train_func=train_func)


def create_config(mapping, config, model):
    resnet_config = config
    to_train = []
    for layer, m in model.named_modules():
        if type(m) == nn.BatchNorm2d or type(m) == nn.Conv2d or type(m) == nn.Linear:
            if layer not in mapping:
                print(layer)
                to_train.append(layer)
            else:
                cornet = mapping[layer]
                if cornet in config:
                    resnet_config[layer] = cornet
                if any(layer in cornet for layer in config['layers']):
                    to_train.append(layer)
                elif type(m) == nn.BatchNorm2d:
                    base = conv_to_norm[cornet]
                    if base in config:
                        resnet_config[layer] = cornet
                    if any(layer in base for layer in config['layers']):
                        to_train.append(layer)
                else:
                    print(f'not trained :{layer}')
        else:
            print(f'Nothing {layer}')
    resnet_config['layers'] = to_train
    return resnet_config
