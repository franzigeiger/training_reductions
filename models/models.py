import functools
import importlib
import logging
import os

import torch
from candidate_models import s3
from candidate_models.base_models.cornet import TemporalPytorchWrapper
from candidate_models.model_commitments.cornets import CORnetCommitment
from model_tools.activations import PytorchWrapper
from submission.utils import UniqueKeyDict
from torch.nn import Module

_logger = logging.getLogger(__name__)


def cornet(identifier, init_weights=True, function=None):
    cornet_type = 'S'
    mod = importlib.import_module(f'cornet.cornet_{cornet_type.lower()}')
    model_ctr = getattr(mod, f'CORnet_{cornet_type.upper()}')
    model = model_ctr()

    if init_weights:
        WEIGHT_MAPPING = {
            'S': 'cornet_s_epoch43.pth.tar'
        }

        class Wrapper(Module):
            def __init__(self, model):
                super(Wrapper, self).__init__()
                self.module = model

        model = Wrapper(model)  # model was wrapped with DataParallel, so weights require `module.` prefix
        framework_home = os.path.expanduser(os.getenv('CM_HOME', '~/.candidate_models'))
        weightsdir_path = os.getenv('CM_TSLIM_WEIGHTS_DIR', os.path.join(framework_home, 'model-weights', 'cornet'))
        weights_path = os.path.join(weightsdir_path, WEIGHT_MAPPING[cornet_type.upper()])
        if not os.path.isfile(weights_path):
            _logger.debug(f"Downloading weights for {identifier} to {weights_path}")
            os.makedirs(weightsdir_path, exist_ok=True)
            s3.download_file(WEIGHT_MAPPING[cornet_type.upper()], weights_path, bucket='cornet-models')
        checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)  # map onto cpu
        model.load_state_dict(checkpoint['state_dict'])
        model = model.module  # unwrap
    if function:
        print('>>>run with function ', function)
        model = function(model)
    from model_tools.activations.pytorch import load_preprocess_images
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = TemporalPytorchWrapper(identifier='%s_%s' % ('CORnet-S', identifier), model=model,
                                     preprocessing=preprocessing,
                                     separate_time=True)
    wrapper.image_size = 224
    return wrapper


def _build_time_mappings(time_mappings):
    return {region: {
        timestep: (time_start + timestep * time_step_size,
                   time_start + (timestep + 1) * time_step_size)
        for timestep in range(0, timesteps)}
        for region, (time_start, time_step_size, timesteps) in time_mappings.items()}


def cornet_s_brainmodel(identifier='', init_weigths=True, function=None):
    identifier = '%s_%s' % ('CORnet-S', identifier)
    # map region -> (time_start, time_step_size, timesteps)
    time_mappings = {
        'V1': (50, 100, 1),
        'V2': (70, 100, 2),
        # 'V2': (20, 50, 2),  # MS: This follows from the movshon anesthesized-monkey recordings, so might not hold up
        'V4': (90, 50, 4),
        'IT': (100, 100, 2),
    }
    return CORnetCommitment(identifier=identifier,
                            activations_model=cornet(identifier=identifier, init_weights=init_weigths,
                                                     function=function),
                            layers=['V1.output-t0'] +
                                   [f'{area}.output-t{timestep}'
                                    for area, timesteps in [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                                    for timestep in timesteps] +
                                   ['decoder.avgpool-t0'],
                            time_mapping=_build_time_mappings(time_mappings))


def alexnet(identifier, init_weights=True, function=None):
    pytorch_model('alexnet', '%s_%s' % ('alexnet', identifier), 224, init_weights, function)


def densenet(identifier, init_weights=True, function=None):
    pytorch_model('densenet169', '%s_%s' % ('densenet', identifier), 224, init_weights, function)


def resnet(identifier, init_weights=True, function=None):
    pytorch_model('resnet101', '%s_%s' % ('resnet', identifier), 224, init_weights, function)


def pytorch_model(function, identifier, image_size, init_weights=True, transformation=None):
    module = importlib.import_module(f'torchvision.models')
    model_ctr = getattr(module, function)
    if transformation:
        model_ctr = transformation(model_ctr)
    from model_tools.activations.pytorch import load_preprocess_images
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    wrapper = PytorchWrapper(identifier=identifier, model=model_ctr(pretrained=init_weights),
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
