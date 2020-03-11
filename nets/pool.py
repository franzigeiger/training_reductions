from brainscore.submission.ml_pool import MLBrainPool
from brainscore.submission.utils import UniqueKeyDict
from brainscore.utils import LazyLoad

from nets import test_models as model_file
from nets.test_models import cornet_s_brainmodel, model_layers
from transformations.configurable import apply_nullify_small, apply_nullify_high, apply_low_variance_cut, \
    apply_high_variance_cut, apply_overflow_weights
from transformations.layer_based import *
from transformations.model_based import apply_incremental_init, apply_fit_std_function


class BaseModelPool(UniqueKeyDict):
    """
    Provides a set of standard models.
    Each entry maps from `name` to an activations extractor.
    """

    def __init__(self):
        super(BaseModelPool, self).__init__()
        self._accessed_base_models = set()

        self._key_functions = {}

        for model in ['alexnet', 'densenet169', 'resnet101']:
            model_func = getattr(model_file, model)
            self.add_model_to_maps(model, '', lambda bound_func=model_func: bound_func('', True))
            self.add_model_to_maps(model, '_random', lambda bound_func=model_func: bound_func('random', False))
            self.add_model_to_maps(model, '_norm_dist',
                                   lambda bound_func=model_func: bound_func('norm_dist', True, apply_norm_dist))
            self.add_model_to_maps(model, '_jumbler',
                                   lambda bound_func=model_func: bound_func('jumbler', True, apply_all_jumbler))
            self.add_model_to_maps(model, '_kernel_jumbler',
                                   lambda bound_func=model_func: bound_func('kernel_jumbler', True,
                                                                            apply_in_kernel_jumbler))
            self.add_model_to_maps(model, '_channel_jumbler',
                                   lambda bound_func=model_func: bound_func('channel_jumbler', True,
                                                                            apply_channel_jumbler))
            self.add_model_to_maps(model, '_norm_dist_kernel',
                                   lambda bound_func=model_func: bound_func('norm_dist_kernel', True,
                                                                            apply_norm_dist_kernel))

        for identifier, function in self._key_functions.items():
            self[identifier] = LazyLoad(function)

    def add_model_to_maps(self, model, suffix, func):
        identifier = f'{model}{suffix}'
        self._key_functions[identifier] = func
        if suffix is not '':
            model_layers[identifier] = model_layers[model]


brain_models = {'CORnet-S': LazyLoad(lambda: cornet_s_brainmodel('base', True)),
                'CORnet-S_random': LazyLoad(lambda: cornet_s_brainmodel('random', True)),
                'CORnet-S_train_IT_seed_0_epoch_10': LazyLoad(
                    lambda: cornet_s_brainmodel('CORnet-S_train_IT_seed_0_epoch_10', True)),
                'CORnet-S_random_2': LazyLoad(lambda: cornet_s_brainmodel('random', False)),
                'CORnet-S_norm_dist': LazyLoad(lambda: cornet_s_brainmodel('norm_dist', True, apply_norm_dist)),
                'CORnet-S_uniform_dist': LazyLoad(
                    lambda: cornet_s_brainmodel('uniform_dist', True, apply_uniform_dist)),
                'CORnet-S_jumbler': LazyLoad(lambda: cornet_s_brainmodel('jumbler', True, apply_all_jumbler)),
                'CORnet-S_kernel_jumbler': LazyLoad(
                    lambda: cornet_s_brainmodel('kernel_jumbler', True, apply_in_kernel_jumbler)),
                'CORnet-S_channel_jumbler': LazyLoad(
                    lambda: cornet_s_brainmodel('channel_jumbler', True, apply_channel_jumbler)),
                'CORnet-S_norm_dist_kernel': LazyLoad(
                    lambda: cornet_s_brainmodel('norm_dist_kernel', True, apply_norm_dist_kernel)),
                'CORnet-S_kaiming': LazyLoad(
                    lambda: cornet_s_brainmodel('kaiming', False, apply_kaiming)),
                f'CORnet-S_incremental_init': LazyLoad(
                    lambda: cornet_s_brainmodel(f'incremental_init', False,
                                                function=apply_incremental_init,
                                                type='custom')),
                f'CORnet-S_trained_incremental_init': LazyLoad(
                    lambda: cornet_s_brainmodel(f'trained_incremental_init', True,
                                                function=apply_incremental_init,
                                                type='custom')),
                f'CORnet-S_std_function_2': LazyLoad(
                    lambda: cornet_s_brainmodel(f'trained_incremental_init', True,
                                                function=apply_fit_std_function,
                                                config={'config': [2]},
                                                type='custom'))
                }

for percent in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    brain_models[f'CORnet-S_low_zero_{percent}'] = LazyLoad(
        lambda percent_fix=percent: cornet_s_brainmodel(f'low_zero_{percent_fix}', False, apply_nullify_small,
                                                        config=[percent_fix]))
    brain_models[f'CORnet-S_high_zero_{percent}'] = LazyLoad(
        lambda percent_fix=percent: cornet_s_brainmodel(f'high_zero_{percent_fix}', False, apply_nullify_high,
                                                        config=[percent_fix]))
    brain_models[f'CORnet-S_low_variance_{percent}'] = LazyLoad(
        lambda percent_fix=percent: cornet_s_brainmodel(f'low_variance_{percent_fix}', False, apply_low_variance_cut,
                                                        config=[percent_fix]))
    brain_models[f'CORnet-S_high_variance_{percent}'] = LazyLoad(
        lambda percent_fix=percent: cornet_s_brainmodel(f'high_variance_{percent_fix}', False, apply_high_variance_cut,
                                                        config=[percent_fix]))
    brain_models[f'CORnet-S_trained_low_zero_{percent}'] = LazyLoad(
        lambda percent_fix=percent: cornet_s_brainmodel(f'trained_low_zero_{percent_fix}', True, apply_nullify_small,
                                                        config=[percent_fix]))
    brain_models[f'CORnet-S_trained_high_zero_{percent}'] = LazyLoad(
        lambda percent_fix=percent: cornet_s_brainmodel(f'trained_high_zero_{percent_fix}', True, apply_nullify_high,
                                                        config=[percent_fix]))
    brain_models[f'CORnet-S_trained_low_variance_{percent}'] = LazyLoad(
        lambda percent_fix=percent: cornet_s_brainmodel(f'trained_low_variance_{percent_fix}', True,
                                                        apply_low_variance_cut,
                                                        config=[percent_fix]))
    brain_models[f'CORnet-S_trained_high_variance_{percent}'] = LazyLoad(
        lambda percent_fix=percent: cornet_s_brainmodel(f'trained_high_variance_{percent_fix}', True,
                                                        apply_high_variance_cut,
                                                        config=[percent_fix]))

for level in [1, 2, 3, 4]:
    brain_models[f'CORnet-S_std_function_{level}'] = LazyLoad(
        lambda level_fix=level: cornet_s_brainmodel(f'trained_incremental_init', True,
                                                    function=apply_norm_dist,
                                                    config={'func': apply_fit_std_function, 'config': [level_fix]},
                                                    type='custom'))

for percent in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
    brain_models[f'CORnet-S_overflow_{percent}'] = LazyLoad(
        lambda percent_fix=percent: cornet_s_brainmodel(f'overflow_{percent_fix}', False, apply_overflow_weights,
                                                        config=[percent_fix]))

base_model_pool = BaseModelPool()

brain_translated_pool = UniqueKeyDict()

ml_brain_pool = MLBrainPool(base_model_pool, model_layers)

for identifier, model in ml_brain_pool.items():
    brain_translated_pool[identifier] = model

for identifier, model in brain_models.items():
    brain_translated_pool[identifier] = model
