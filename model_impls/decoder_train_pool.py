from brainscore.utils import LazyLoad
from submission.utils import UniqueKeyDict

from model_impls.test_models import cornet_s_brainmodel
from transformations.layer_based import *

brain_models = {
    # 'CORnet-S': LazyLoad(lambda: cornet_s_brainmodel('base', True)),
    'CORnet-S_train_random': LazyLoad(lambda: cornet_s_brainmodel('train_random', False, train_decoder=True)),
    'CORnet-S_train_norm_dist': LazyLoad(
        lambda: cornet_s_brainmodel('train_norm_dist', True, apply_norm_dist, train_decoder=True)),
    # 'CORnet-S_train_uniform_dist': LazyLoad(lambda: cornet_s_brainmodel('uniform_dist', True, apply_uniform_dist, train_decoder=True)),
    'CORnet-S_train_jumbler': LazyLoad(
        lambda: cornet_s_brainmodel('train_jumbler', True, apply_all_jumbler, train_decoder=True)),
    'CORnet-S_train_kernel_jumbler': LazyLoad(
        lambda: cornet_s_brainmodel('train_kernel_jumbler', True, apply_in_kernel_jumbler, train_decoder=True)),
    'CORnet-S_train_channel_jumbler': LazyLoad(
        lambda: cornet_s_brainmodel('train_channel_jumbler', True, apply_channel_jumbler, train_decoder=True)),
    'CORnet-S_train_norm_dist_kernel': LazyLoad(
        lambda: cornet_s_brainmodel('train_norm_dist_kernel', True, apply_norm_dist_kernel, train_decoder=True)),
    'CORnet-S_train_IT_random': LazyLoad(
        lambda: cornet_s_brainmodel('train_IT_random', False, train_decoder=True, config=['IT', 'decoder'])),
    'CORnet-S_train_IT_norm_dist': LazyLoad(
        lambda: cornet_s_brainmodel('train_IT_norm_dist', True, apply_norm_dist, train_decoder=True,
                                    config=['IT', 'decoder'])),
    # 'CORnet-S_train_uniform_dist': LazyLoad(lambda: cornet_s_brainmodel('uniform_dist', True, apply_uniform_dist, train_decoder=True)),
    'CORnet-S_train_IT_jumbler': LazyLoad(
        lambda: cornet_s_brainmodel('train_IT_jumbler', True, apply_all_jumbler, train_decoder=True,
                                    config=['IT', 'decoder'])),
    'CORnet-S_train_IT_kernel_jumbler': LazyLoad(
        lambda: cornet_s_brainmodel('train_IT_kernel_jumbler', True, apply_in_kernel_jumbler, train_decoder=True,
                                    config=['IT', 'decoder'])),
    'CORnet-S_train_IT_channel_jumbler': LazyLoad(
        lambda: cornet_s_brainmodel('train_IT_channel_jumbler', True, apply_channel_jumbler, train_decoder=True,
                                    config=['IT', 'decoder'])),
    'CORnet-S_train_IT_norm_dist_kernel': LazyLoad(
        lambda: cornet_s_brainmodel('train_IT_norm_dist_kernel', True, apply_norm_dist_kernel, train_decoder=True,
                                    config=['IT', 'decoder'])),
}

brain_translated_pool = UniqueKeyDict()

for identifier, model in brain_models.items():
    brain_translated_pool[identifier] = model
