from brainscore.utils import LazyLoad
from candidate_models.base_models import pytorch_model
from cornet import cornet_s
from submission.ml_pool import MLBrainPool
from submission.utils import UniqueKeyDict

from model_impls import test_models as model_file
from model_impls.test_models import cornet_s_brainmodel, model_layers
from perturbations.model_based import *


class BaseModelPool(UniqueKeyDict):
    """
    Provides a set of standard models.
    Each entry maps from `name` to an activations extractor.
    """

    def __init__(self):
        super(BaseModelPool, self).__init__()
        self._accessed_base_models = set()

        _key_functions = {}

        for model in ['alexnet', 'densenet169', 'resnet101']:
            model_func = getattr(model_file, model)
            _key_functions[f'{model}'] = lambda bound_func=model_func: bound_func('', True)
            _key_functions[f'{model}_random'] = lambda bound_func=model_func: bound_func('random', False)
            _key_functions[f'{model}_norm_dist'] = lambda bound_func=model_func: bound_func('norm_dist', True, apply_norm_dist)
            _key_functions[f'{model}_uniform_dist'] = lambda bound_func=model_func: bound_func('uniform_dist', True, apply_uniform_dist)
            _key_functions[f'{model}_jumbler'] = lambda bound_func=model_func: bound_func('jumbler', True, apply_jumbler)

        for identifier, function in _key_functions.items():
            self[identifier] = LazyLoad(function)


brain_models = {
    'CORnet-S': LazyLoad(lambda: cornet_s_brainmodel('base', True)),
    'CORnet-S_random': LazyLoad(lambda: cornet_s_brainmodel('random', False)),
    'CORnet-S_norm_dist': LazyLoad(lambda: cornet_s_brainmodel('norm_dist', True, apply_norm_dist)),
    'CORnet-S_uniform_dist': LazyLoad(lambda: cornet_s_brainmodel('uniform_dist', True, apply_uniform_dist)),
    'CORnet-S_jumbler': LazyLoad(lambda: cornet_s_brainmodel('jumbler', True, apply_jumbler)),
}

base_model_pool = BaseModelPool()

brain_translated_pool = UniqueKeyDict()

ml_brain_pool = MLBrainPool(base_model_pool, model_layers)

for identifier, model in ml_brain_pool.items():
    brain_translated_pool[identifier] = model

for identifier, model in brain_models.items():
    brain_translated_pool[identifier] = model
