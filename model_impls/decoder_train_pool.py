from brainscore.utils import LazyLoad
from submission.utils import UniqueKeyDict

from model_impls.test_models import cornet_s_brainmodel
from transformations.layer_based import *

brain_models = {
    # 'CORnet-S': LazyLoad(lambda: cornet_s_brainmodel('base', True)),
    'CORnet-S_train_random': LazyLoad(lambda: cornet_s_brainmodel('random', False, train_decoder=True)),
    'CORnet-S_train_norm_dist': LazyLoad(lambda: cornet_s_brainmodel('norm_dist', True, apply_norm_dist, train_decoder=True)),
    # 'CORnet-S_train_uniform_dist': LazyLoad(lambda: cornet_s_brainmodel('uniform_dist', True, apply_uniform_dist, train_decoder=True)),
    'CORnet-S_train_jumbler': LazyLoad(lambda: cornet_s_brainmodel('jumbler', True, apply_all_jumbler, train_decoder=True)),
    'CORnet-S_train_kernel_jumbler': LazyLoad(lambda: cornet_s_brainmodel('kernel_jumbler', True, apply_in_kernel_jumbler, train_decoder=True)),
    'CORnet-S_train_channel_jumbler': LazyLoad(lambda: cornet_s_brainmodel('channel_jumbler', True, apply_channel_jumbler, train_decoder=True)),
    'CORnet-S_train_norm_dist_kernel': LazyLoad(
        lambda: cornet_s_brainmodel('norm_dist_kernel', True, apply_norm_dist_kernel)),
}

brain_translated_pool = UniqueKeyDict()

for identifier, model in brain_models.items():
    brain_translated_pool[identifier] = model
