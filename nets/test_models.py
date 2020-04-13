import functools
import importlib
import logging
import os

import numpy as np
import torch
import torchvision
from brainscore.submission.utils import UniqueKeyDict
from candidate_models.base_models.cornet import TemporalPytorchWrapper
from candidate_models.model_commitments.cornets import CORnetCommitment
from model_tools.activations import PytorchWrapper
from model_tools.utils import s3
from torch.nn import Module

from nets.trainer import output_path
from nets.trainer import train
from transformations.layer_based import apply_to_net
from transformations.model_based import apply_to_one_layer

_logger = logging.getLogger(__name__)

seed = 0
batch_fix = False


# def cornet_long(identifier, init_weights=True, type='layer', function=None, config=None):
#
#     cornet(identifier, init_weights, else config + {'model_func': function})

def cornet(identifier, init_weights=True, config=None):
    _logger.info('Run normal benchmark')
    # if batch_fix:
    #     identifier = f'{identifier}_BF'
    model = get_model(identifier, init_weights, config)
    # for name, m in model.named_modules():
    #     if type(m) == nn.BatchNorm2d and m.num_batches_tracked.data.cpu() == 0:
    #         m.momentum = 0
    from model_tools.activations.pytorch import load_preprocess_images
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = TemporalPytorchWrapper(identifier=identifier, model=model,
                                     preprocessing=preprocessing,
                                     separate_time=True)
    wrapper.image_size = 224
    return wrapper


def get_model(identifier, init_weights=True, config=None):
    if config is None:
        config = {}
    print(f'Configuration: {config}')
    cornet_type = 'S'
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if 'batchnorm' in config or batch_fix:
    #     print('We load the alternative batchnorm model!')
    #     mod = importlib.import_module(f'cornet.cornet_s_compression')
    #     identifier = f'{identifier}_BF'
    # else:
    mod = importlib.import_module(f'cornet.cornet_{cornet_type.lower()}')
    model_ctr = getattr(mod, f'CORnet_{cornet_type.upper()}')
    model = model_ctr()
    if init_weights:
        _logger.info('Initialize weights')
        # if batch_fix:
        #     checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)
        #     set_running_averages(checkpoint)
        #     # set_half_running_averages(checkpoint, config)
        #     # delete_running_averages(checkpoint)
        # else:
        class Wrapper(Module):
            def __init__(self, model):
                super(Wrapper, self).__init__()
                self.module = model

        model = Wrapper(model)  # model was wrapped with DataParallel, so weights require `module.` prefix
        weights_path = get_weights(identifier)
        checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except:
            model.module.load_state_dict(checkpoint['state_dict'])
        model = model.module  # unwrap
    if batch_fix:
        config['batchnorm'] = True
    if 'model_func' in config or 'layer_func' in config:
        _logger.info('Apply function')
        if 'model_func' in config:
            print('>>>run with function ', config['model_func'])
            model = config['model_func'](model, config)
        else:
            print('>>>run with net function and add', config)
            model = apply_to_net(model, config)
    return model


def get_resnet50(init_weights=True):
    module = importlib.import_module(f'torchvision.models')
    torchvision.models.resnet50()
    model_ctr = getattr(module, 'resnet50')
    return model_ctr(pretrained=init_weights)

def get_weights(identifier):
    if identifier == 'CORnet-S_base':
        WEIGHT_MAPPING = {
            'S': 'cornet_s_epoch43.pth.tar'
        }
        cornet_type = 'S'
        framework_home = os.path.expanduser(os.getenv('CM_HOME', '~/.candidate_models'))
        weightsdir_path = os.getenv('CM_TSLIM_WEIGHTS_DIR', os.path.join(framework_home, 'model-weights', 'cornet'))
        weights_path = os.path.join(weightsdir_path, WEIGHT_MAPPING[cornet_type.upper()])
        if not os.path.isfile(weights_path):
            _logger.debug(f"Downloading weights for {identifier} to {weights_path}")
            os.makedirs(weightsdir_path, exist_ok=True)
            s3.download_file(WEIGHT_MAPPING[cornet_type.upper()], weights_path, bucket='cornet-models')
    else:
        weights_path = output_path + f'{identifier}.pth.tar'
    return weights_path


def run_model_training(model, identifier, config=None, train_func=None):
    if batch_fix:
        identifier = f'{identifier}_BF'
    if config is None:
        config = {'layers': ['decoder']}
    if 'full' not in config:
        for name, m in model.named_parameters():
            # if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Linear or type(m) == torch.nn.BatchNorm2d:
            if any(value in name for value in config['layers']):
                m.requires_grad = True
            else:
                m.requires_grad = False
    # model.decoder.requires_grad = True
    if train_func:
        model = train_func(identifier, model)
    else:
        model = train(identifier, model)
    return model


def _build_time_mappings(time_mappings):
    return {region: {
        timestep: (time_start + timestep * time_step_size,
                   time_start + (timestep + 1) * time_step_size)
        for timestep in range(0, timesteps)}
        for region, (time_start, time_step_size, timesteps) in time_mappings.items()}


def cornet_s_brainmodel_short(identifier='', init_weigths=True, config=None):
    if not identifier.startswith('CORnet-S'):
        identifier = '%s_%s' % ('CORnet-S', identifier)
    # map region -> (time_start, time_step_size, timesteps)
    time_mappings = {
        'V1': (50, 100, 1),
        'V2': (70, 100, 2),
        # 'V2': (20, 50, 2),  # MS: This follows from the movshon anesthesized-monkey recordings, so might not hold up
        'V4': (90, 50, 4),
        'IT': (100, 100, 2),
    }
    model = cornet(identifier=identifier, init_weights=init_weigths, config=config)
    return CORnetCommitment(identifier=model.identifier,
                            activations_model=model,
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                                    for timestep in timesteps] +
                                   ['decoder.avgpool-t0'],
                            time_mapping=_build_time_mappings(time_mappings))


def cornet_s_brainmodel(identifier='', init_weigths=True, function=None, config=None, type='layer'):
    if function and type == 'layer':
        config['layer_func'] = function
    if type == 'model':
        config = config + {'layer_func': function, 'model_func': apply_to_one_layer}
    if type == 'custom':
        config['model_func'] = function
    return cornet_s_brainmodel_short(identifier, init_weigths, config)


def alexnet(identifier, init_weights=True, function=None):
    return pytorch_model('alexnet', '%s_%s' % ('alexnet', identifier), 224, init_weights, function)


def densenet169(identifier, init_weights=True, function=None):
    return pytorch_model('densenet169', '%s_%s' % ('densenet', identifier), 224, init_weights, function)


def resnet101(identifier, init_weights=True, function=None):
    return pytorch_model('resnet101', '%s_%s' % ('resnet', identifier), 224, init_weights, function)


def resnet50(identifier, init_weights=True, function=None):
    return pytorch_model('resnet50', '%s_%s' % ('resnet', identifier), 224, init_weights, function)


def resnet_michael(identifier, init_weights=True, function=None):
    from tbs import load_model
    from tbs.tfkeras_wrapper_for_brainscore import TFKerasWrapper, resnet_preprocessing
    epoch = int(identifier.split('_')[-1])
    _logger.info(f'We load weights from epoch {epoch}')
    if epoch == 0:
        model = load_model.ResNet50(weights='random_weights', batch_size=None, trainable=False)
    else:
        model = load_model.ResNet50(weights='imagenet', epoch=epoch, batch_size=None, trainable=False)
    return TFKerasWrapper(model,
                          preprocessing=resnet_preprocessing,
                          identifier=identifier)


def resnet_michael_layers():
    from tbs.tfkeras_wrapper_for_brainscore import resnet_layers
    return resnet_layers


def pytorch_model(function, identifier, image_size, init_weights=True, transformation=None):
    module = importlib.import_module(f'torchvision.models')
    model_ctr = getattr(module, function)
    model = model_ctr(pretrained=init_weights)
    if transformation:
        model = apply_to_net(model, transformation)
    from model_tools.activations.pytorch import load_preprocess_images
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = PytorchWrapper(identifier=identifier, model=model,
                             preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper


def resnext101_layers():
    return (['conv1'] +
            # note that while relu is used multiple times, by default the last one will overwrite all previous ones
            [f"layer{block + 1}.{unit}.relu"
             for block, block_units in enumerate([3, 4, 23, 3]) for unit in range(block_units)] +
            ['avgpool'])


layers = {
    'alexnet':
        [  # conv-relu-[pool]{1,2,3,4,5}
            'features.2', 'features.5', 'features.7', 'features.9', 'features.12',
            'classifier.2', 'classifier.5'],  # fc-[relu]{6,7,8}
    'densenet169':
        ['features.conv0'] + ['features.pool0'] +
        [f'features.denseblock1.denselayer{i + 1}.conv1' for i in range(6)] +
        [f'features.denseblock1.denselayer{i + 1}.conv2' for i in range(6)] + ['features.transition1'] +
        [f'features.denseblock2.denselayer{i + 1}.conv1' for i in range(12)] +
        [f'features.denseblock2.denselayer{i + 1}.conv2' for i in range(12)] + ['features.transition2'] +
        [f'features.denseblock3.denselayer{i + 1}.conv1' for i in range(24)] +
        [f'features.denseblock3.denselayer{i + 1}.conv2' for i in range(24)] + ['features.transition3'] +
        [f'features.denseblock3.denselayer{i + 1}.conv2' for i in range(16)] +
        [f'features.denseblock3.denselayer{i + 1}.conv2' for i in range(16)] + ['features.norm5'],
    'resnet101': resnext101_layers()
}


class ModelLayers(UniqueKeyDict):
    def __init__(self, layers):
        super(ModelLayers, self).__init__()
        for basemodel_identifier, default_layers in layers.items():
            self[basemodel_identifier] = default_layers

    @staticmethod
    def _item(item):
        # if item.startswith('mobilenet'):
        #     return "_".join(item.split("_")[:2])
        if item.startswith('bagnet'):
            return 'bagnet'
        return item

    def __getitem__(self, item):
        return super(ModelLayers, self).__getitem__(self._item(item))

    def __contains__(self, item):
        return super(ModelLayers, self).__contains__(self._item(item))


model_layers = ModelLayers(layers)
