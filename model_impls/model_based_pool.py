from brainscore.utils import LazyLoad
from submission.utils import UniqueKeyDict
from torch import nn

from model_impls.test_models import cornet_s_brainmodel
# cornet has
from transformations.layer_based import apply_norm_dist, apply_all_jumbler, apply_fixed_value, apply_fixed_value_small

brain_models = {}

def load_single_layer_models():
    model = cornet_s_brainmodel('base', True).activations_model._model
    layer_number = 0
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            layer_number = layer_number + 1
            brain_models[f'CORnet-S_norm_dist_L{layer_number}'] = LazyLoad(
                lambda layer_number=layer_number: cornet_s_brainmodel(f'norm_dist_L{layer_number}', True, function=apply_norm_dist, config=[layer_number],
                                            type='model'))
            brain_models[f'CORnet-S_jumbler_L{layer_number}'] = LazyLoad(
                lambda layer_number=layer_number: cornet_s_brainmodel(f'jumbler_L{layer_number}', True, function=apply_all_jumbler, config=[layer_number],
                                            type='model'))
            brain_models[f'CORnet-S_fixed_value_L{layer_number}'] = LazyLoad(
                lambda layer_number=layer_number: cornet_s_brainmodel(f'fixed_value_L{layer_number}', True, function=apply_fixed_value, config=[layer_number],
                                                                      type='model'))
            brain_models[f'CORnet-S_fixed_value_small_L{layer_number}'] = LazyLoad(
                lambda layer_number=layer_number: cornet_s_brainmodel(f'fixed_value_small_L{layer_number}', True,
                                                                      function=apply_fixed_value_small,
                                                                      config=[layer_number],
                                                                      type='model'))


load_single_layer_models()

brain_translated_pool = UniqueKeyDict()

for identifier, model in brain_models.items():
    brain_translated_pool[identifier] = model
