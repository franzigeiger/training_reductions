from brainscore.utils import LazyLoad
from submission.utils import UniqueKeyDict
from torch import nn

from model_impls.test_models import cornet_s_brainmodel

# cornet has
from transformations.layer_based import apply_norm_dist, apply_all_jumbler

brain_models = {}

def load_single_layer_models():
    model = cornet_s_brainmodel('base', True).activations_model._model
    layer_number = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d or ((type(m) == nn.Linear or type(m) == nn.BatchNorm2d)):
            layer_number += 1
            brain_models[f'CORnet-S_norm_dist_L{layer_number}'] = LazyLoad(
                lambda: cornet_s_brainmodel(f'norm_dist_L{layer_number}', True, apply_norm_dist, config=(layer_number),
                                            type='model')),
            brain_models[f'CORnet-S_jumbler_L{layer_number}'] = LazyLoad(
                lambda: cornet_s_brainmodel(f'jumbler_L{layer_number}', True, apply_all_jumbler, config=(layer_number),
                                            type='model')),


load_single_layer_models()

brain_translated_pool = UniqueKeyDict()

for identifier, model in brain_models.items():
    brain_translated_pool[identifier] = model
