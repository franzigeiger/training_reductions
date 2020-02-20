from nets.test_models import run_model_training, get_model
from transformations.layer_based import *
from transformations.model_based import *

trained_models = {
    'CORnet-S_train_gabor_reshape_no_batchnorm': {'model_func': apply_gabors,
                                                  'layers': ['V1.conv2', 'V1.norm2', 'V2', 'V4', 'IT', "decoder"],
                                                  'reshape': True, 'epochs': 10},
    'CORnet-S_train_gabor_reshape': {'model_func': apply_gabors,
                                     'layers': ['V1.conv2', 'V1.norm1', 'V1.norm2', 'V2', 'V4', 'IT', "decoder"],
                                     'reshape': True, },
    'CORnet-S_train_gabor_no_batchnorm': {'model_func': apply_gabors,
                                          'layers': ['V1.conv2', 'V1.norm2', 'V2', 'V4', 'IT', "decoder"],
                                          'reshape': False},
    'CORnet-S_train_gabor': {'model_func': apply_gabors,
                             'layers': ['V1.conv2', 'V1.norm1', 'V1.norm2', 'V2', 'V4', 'IT', "decoder"],
                             'reshape': False},
    'CORnet-S_train_second_no_batchnorm': {'model_func': apply_second_layer,
                                           'layers': ['V2', 'V4', 'IT', "decoder"],
                                           'reshape': True},
    'CORnet-S_train_second': {'model_func': apply_second_layer,
                              'layers': ['V2', 'V4', 'IT', "decoder"]},

    'CORnet-S_train_second_corr_only': {'model_func': apply_second_layer_only,
                                        'layers': ['V2', 'V4', 'IT', "decoder"], },
    'CORnet-S_train_gabor_fit_second_corr': {'model_func': apply_gabor_fit_second_layer,
                                             'layers': ['V2', 'V4', 'IT', "decoder"],
                                             'reshape': False,
                                             'file': '/braintree/home/fgeiger/weight_initialization/gabors_tiago_2.npy'},
    'CORnet-S_train_gabor_fit': {'model_func': apply_gabors_fit,
                                 'layers': ['V1.conv2', 'V1.norm2', 'V2', 'V4', 'IT', "decoder"],
                                 'reshape': False,
                                 'file': '/braintree/home/fgeiger/weight_initialization/gabors_tiago_2.npy'},
    'CORnet-S_train_gabor_dist': {'model_func': apply_gabors_dist_old,
                                  'layers': ['V1.conv2', 'V1.norm2', 'V2', 'V4', 'IT', "decoder"],
                                  'reshape': False,
                                  'file': '/braintree/home/fgeiger/weight_initialization/gabors_tiago_2.npy'},

    'CORnet-S_rand_conv1': {'model_func': apply_gabors,
                            'layers': ['V1.conv2', 'V1.norm2', 'V2', 'V4', 'IT', "decoder"],
                            'reshape': True, },
    # Round 3
    'CORnet-S_train_second_corr_no_resize': {'model_func': apply_second_layer_corr_no_reshape,
                                             'layers': ['V2', 'V4', 'IT', "decoder"]},
    'CORnet-S_train_gabor_fit_second_corr_no_resize': {'model_func': apply_gabor_fit_second_layer_no_reshape,
                                                       'layers': ['V2', 'V4', 'IT', "decoder"],
                                                       'reshape': False,
                                                       'file': '/braintree/home/fgeiger/weight_initialization/gabors_tiago_2.npy'},
    'CORnet-S_train_gabor_fit_second_kernel_conv': {'model_func': apply_first_fit_kernel_convolution_second_layer,
                                                    'layers': ['V2', 'V4', 'IT', "decoder"],
                                                    'reshape': False,
                                                    'file': '/braintree/home/fgeiger/weight_initialization/gabors_tiago_2.npy'},
    'CORnet-S_train_second_kernel_conv': {'model_func': apply_kernel_convolution_second_layer,
                                          'layers': ['V2', 'V4', 'IT', "decoder"]},

    # round 4 ditribution init
    'CORnet-S_train_gabor_dist_second_corr_no_resize': {'model_func': apply_gabor_dist_second_layer_no_reshape,
                                                        'layers': ['V2', 'V4', 'IT', "decoder"],
                                                        'reshape': False,
                                                        'file': '/braintree/home/fgeiger/weight_initialization/gabors_tiago_scaled_cornet_2.npy'},
    'CORnet-S_train_gabor_multi_dist': {'model_func': apply_gabors_dist,
                                        'layers': ['V1.conv2', 'V1.norm2', 'V2', 'V4', 'IT', "decoder"],
                                        'reshape': False,
                                        'file': '/braintree/home/fgeiger/weight_initialization/gabors_tiago_scaled_cornet_2.npy'},
    'CORnet-S_train_gabor_dist_second_kernel_conv': {'model_func': apply_first_dist_kernel_convolution_second_layer,
                                                     'layers': ['V2', 'V4', 'IT', "decoder"],
                                                     'reshape': False,
                                                     'file': '/braintree/home/fgeiger/weight_initialization/gabors_tiago_scaled_cornet_2.npy'},
    'CORnet-S_train_gabor_scrumble': {'model_func': apply_gabors_scrumble,
                                      'layers': ['V1.conv2', 'V1.norm2', 'V2', 'V4', 'IT', "decoder"],
                                      'file': '/braintree/home/fgeiger/weight_initialization/gabors_tiago_scaled_cornet_2.npy'},
    'CORnet-S_train_random': {},
    'CORnet-S_train_all': {'layers': ['V1', 'V2', 'V4', 'IT', "decoder"]},
    'CORnet-S_full': {'layers': ['V1', 'V2', 'V4', 'IT', "decoder"]},
    'CORnet-S_train_norm_dist': {'layer_func': apply_norm_dist},
    'CORnet-S_train_jumbler': {'layer_func': apply_all_jumbler},
    'CORnet-S_train_kernel_jumbler': {'layer_func': apply_in_kernel_jumbler},
    'CORnet-S_train_channel_jumbler': {'layer_func': apply_channel_jumbler},
    'CORnet-S_train_norm_dist_kernel': {'layer_func': apply_norm_dist_kernel},
    'CORnet-S_train_IT_random': {'layers': ['IT', 'decoder']},
    'CORnet-S_train_IT_seed_0': {'layers': ['IT', 'decoder']},
    'CORnet-S_train_IT_norm_dist': {'layers': ['IT', 'decoder'], 'layer_func': apply_norm_dist},
    'CORnet-S_train_IT_jumbler': {'layers': ['IT', 'decoder'], 'layer_func': apply_all_jumbler},
    'CORnet-S_train_IT_kernel_jumbler': {'layers': ['IT', 'decoder'], 'layer_func': apply_in_kernel_jumbler},
    'CORnet-S_train_IT_channel_jumbler': {'layers': ['IT', 'decoder'], 'layer_func': apply_channel_jumbler},
    'CORnet-S_train_IT_norm_dist_kernel': {'layers': ['IT', 'decoder'], 'layer_func': apply_norm_dist_kernel}
}


def train_model(model, train_func=None):
    assert trained_models[model] is not None
    run_model_training(identifier=model, init_weights=False, config=trained_models[model], train_func=train_func)


if __name__ == '__main__':
    model = 'CORnet-S_train_gabor_scrumble'
    model = get_model(model, False, trained_models[model])
    assert model is not None
