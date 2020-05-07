benchmarks = ['movshon.FreemanZiemba2013.V1-pls',
              'movshon.FreemanZiemba2013.V2-pls',
              'dicarlo.Majaj2015.V4-pls',
              'dicarlo.Majaj2015.IT-pls',
              'dicarlo.Rajalingham2018-i2n',
              'fei-fei.Deng2009-top1']

benchmarks_public = ['movshon.FreemanZiemba2013public.V1-pls',
                     'movshon.FreemanZiemba2013public.V2-pls',
                     'dicarlo.Majaj2015public.V4-pls',
                     'dicarlo.Majaj2015public.IT-pls',
                     'dicarlo.Rajalingham2018public-i2n',
                     'fei-fei.Deng2009-top1']

seed = 0

best_models_imagenet = {
    'CORnet-S_full': 'full',
    'CORnet-S_train_gabor_multi_dist': 'V1.conv1',
    'CORnet-S_train_gabor_dist_weight_dist_channel': 'V1.conv2',
    'CORnet-S_train_gmk1_wmc2_kn3': 'V2.conv_input',
    'CORnet-S_train_gmk1_wmc2_kn3_kn4': 'V2.skip',
    'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5': 'V2.conv1',
    'CORnet-S_train_gmk1_wmc2_kn3_kn4_ln5_wm6_v2': 'V2.conv2',
    'CORnet-S_train_gmk1_wmc2_kn3_kn4_kn5_wmc6_kn7_v2': 'V2.conv3',
    'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12': 'V4.conv2',
}
best_models_brain = {
    'CORnet-S_full': 'full',
    'CORnet-S_train_gabor_multi_dist': 'V1.conv1',
    'CORnet-S_train_gabor_dist_weight_dist_kernel': 'V1.conv2',
    'CORnet-S_train_gmk1_gmk2_kn3_kn4': 'V2.skip',
    'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5': 'V2.conv1',
    'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_1_gpu': 'V2.conv2',
    'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_bdk7': 'V2.conv3',
    'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12': 'V4.conv2',
}

best_models_brain_avg_all = {
    # 'CORnet-S_full': 'Full',
    # 'CORnet-S_brain3_kn8_kn9_kn10_kn11_kn12_tra_bi' : 'V4.conv3_special',
    # 'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_tr_bi' : 'V4.conv3_special',
    # 'CORnet-S_brain3_knall_IT_bi': 'IT.conv3',
    # 'CORnet-S_brain3_t7_t12_knall_IT_bi' : 'IT.conv3_special', old
    # 'CORnet-S_brain_t7_t12_knk15_IT_bi' : 'IT.conv3_special',
    'CORnet-S_train_gmk1_bd2_bi': 'V1.conv2',
    # 'CORnet-S_train_gmk1_gmk2_ln3_bi': 'V2.input',
    # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_bi': 'V2.skip',
    'CORnet-S_train_gmk1_gmk2_ln3_kn4_ln5_bi': 'V2.conv1',
    # 'CORnet-S_train_gmk1_gmk2_kn3_mi4_kn5_bd6_bi': 'V2.conv2',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_bi': 'V2.conv3_special',
    'CORnet-S_train_gmk1_gmk2_kn3_mi4_kn5_bd6_kn7_bi': 'V2.conv3',
    'CORnet-S_brain3_kn8_kn9_kn10_kn11_kn12_bi': 'V4.conv3',
    'CORnet-S_brain_wmc15_IT_bi': 'IT.conv3',
}
best_special_brain = {
    'CORnet-S_brain_t7_t12_wmc15_IT_bi': 'IT.conv3_special',
    'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_tr_bi': 'V4.conv3_special',
    'CORnet-S_train_gmk1_gmk2_kn3_mi4_kn5_bd6_bi': 'V2.conv3_special',
}
# old
best_models_brain_init = {
    'CORnet-S_brain3_kn8_kn9_kn10_kn11_kn12_tra_bi': 'V4.conv3_special',
    'CORnet-S_brain3_knall_IT.conv2_bi': 'IT.conv3',
    'CORnet-S_brain3_knall_IT_bi': 'IT.conv3',
    'CORnet-S_brain3_t7_t12_knall_IT_bi': 'IT.conv3_special',
    'CORnet-S_full': 'Full',
    'CORnet-S_train_gabor_dist_both_kernel_bi': 'V1.conv2',
    'CORnet-S_train_gmk1_gmk2_ln3_bi': 'V2.input',
    'CORnet-S_train_gmk1_gmk2_kn3_ln4_bi': 'V2.skip',
    'CORnet-S_train_gmk1_gmk2_ln3_kn4_ln5_bi': 'V2.conv1',
    'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_bi': 'V2.conv2',
    'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_kn6_kn7_bi': 'V2.conv3',
    'CORnet-S_brain4_kn8_kn9_kn10_kn11_kn12_tra_bi': 'V4.conv2',
    'CORnet-S_brain3_kn8_kn9_kn10_kn11_kn12_bi': 'V4.conv3',
    # 'CORnet-S_train_decoder' : 'IT.conv3',
}

layer_best_2 = {
    'Full': 18,
    'Standard training': 18,
    'V1.conv2': 16,
    'V2.input': 15,
    'V2.skip': 14,
    'V2.conv1': 13,
    'V2.conv2': 12,
    'V2.conv3': 11,
    'V4.input': 10,
    'V4.skip': 9,
    'V4.conv1': 8,
    'V4.conv2': 7,
    'V4.conv3': 6,
    'V4.conv3_special': 8,
    'IT.input': 5,
    'IT.skip': 4,
    'IT.conv1': 3,
    'IT.conv2': 2,
    'IT.conv3': 1,
    'IT.conv3_special': 4,
    'decoder': 0,
    'V2.conv3_special': 12
    # 'CORnet-S_brain2_kn8_kn9_kn10_kn11_bi' : 7,
    # 'CORnet-S_brain2_kn8_kn9_kn10_kn11_kn12_bi' : 6,
    # 'CORnet-S_full': 'Full',
}

layer_best = {
    'CORnet-S_train_kn1_kn2_bi_v2': 16,
    'CORnet-S_train_kn1_kn2_kn3_kn4_kn5_kn6_bi_v2': 12,
    # 'CORnet-S_train_kn1_kn2_kn3_kn4_kn5_kn6_kn7_bi' : 10,
    # 'CORnet-S_brain2_kn8_kn9_kn10_kn11_bi' : 6,
    # 'CORnet-S_brain2_kn8_kn9_kn10_kn11_kn12_bi' : 5,
    # 'CORnet-S_brain2_knall_IT_bi': 1,
    # 'CORnet-S_brain2_t7_kn8_kn9_kn10_kn11_bi' : 6,
    'CORnet-S_brain2_t7_t12_knall_IT_bi_v2': 4,
    # 'CORnet-S_random' : 0
}

layer_random = {
    'CORnet-S_full': 17,
    'CORnet-S_train_V2': 16,
    'CORnet-S_train_V4': 12,
    'CORnet-S_train_IT_seed_0': 6,
    'CORnet-S_train_random': 1,
    'CORnet-S_random': 0
}

random_scores = {
    'CORnet-S_full': 'Standard training',
    'CORnet-S_train_V2': 'V1.conv2',
    'CORnet-S_train_V4': 'V2.conv3',
    'CORnet-S_train_IT_seed_0': 'V4.conv3',
    'CORnet-S_train_random': 'IT.conv3',
    "CORnet-S_random": 'decoder',
}

best_brain_avg = {
    'CORnet-S_full': 'Full',
    # 'CORnet-S_brain3_kn8_kn9_kn10_kn11_kn12_tra_bi' : 'V4.conv3_special',
    # 'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_tr_bi' : 'V4.conv3_special',
    'CORnet-S_brain3_knall_IT.conv2_bi': 'IT.conv3',
    # 'CORnet-S_brain3_knall_IT_bi': 'IT.conv3',
    # 'CORnet-S_brain3_t7_t12_knall_IT_bi' : 'IT.conv3_special', old
    # 'CORnet-S_brain_t7_t12_knk15_IT_bi' : 'IT.conv3_special',
    'CORnet-S_train_gabor_dist_both_kernel_bi_BF': 'V1.conv2',
    # 'CORnet-S_train_gmk1_gmk2_ln3_bi': 'V2.input',
    # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_bi': 'V2.skip',
    'CORnet-S_train_gmk1_gmk2_ln3_kn4_ln5_bi': 'V2.conv1',
    'CORnet-S_train_gmk1_gmk2_kn3_mi4_kn5_bd6_bi': 'V2.conv2',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_bi': 'V2.conv3_special',
    'CORnet-S_train_gmk1_gmk2_kn3_mi4_kn5_bd6_kn7_bi': 'V2.conv3',
    'CORnet-S_brain3_kn8_kn9_kn10_kn11_kn12_bi': 'V4.conv3',
}

convergence_epoch = {
    'CORnet-S_full': 43,
    'CORnet-S_train_V2': 20,
    'CORnet-S_train_V4': 43,
    'CORnet-S_train_IT_seed_0': 37,
    'CORnet-S_train_random': 28,
    'CORnet-S_random': 0,
    'CORnet-S_brain_t7_t12_wmc15_IT_bi': 38,
    'resnet_v1_CORnet-S_full': 32,
    'resnet_v1_CORnet-S_train_random': 24,
    'resnet_v1_CORnet-S_brain_t7_t12_wmc15_bi': 26,
    'resnet_v1_CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wmc6_kn7_tr_bi': 24,
    'alexnet_v1_CORnet-S_brain_t7_t12_wmc15_IT_bi': 33,
    'alexnet_v1_CORnet-S_train_random': 29,
    'alexnet_v1_CORnet-S_full': 36
}
convergence_seed_42 = {
    'CORnet-S_train_V2_seed42': 32,
    'CORnet-S_full_seed42': 29,
    'CORnet-S_train_V4_seed42': 31,
    'CORnet-S_train_IT_seed_0': 40
}

convergence_images = {
    'CORnet-S_full_img500000': 21,
    'CORnet-S_full_img100000': 66,
    'CORnet-S_full_img10000': 35,
    'CORnet-S_full_img1000': 63,
    'CORnet-S_full_img100': 61,
    'CORnet-S_brain_t7_t12_wmc15_IT_bi_img500000': 28,
    'CORnet-S_brain_t7_t12_wmc15_IT_bi_img100000': 41,
    'CORnet-S_brain_t7_t12_wmc15_IT_bi_img10000': 22,
    'CORnet-S_brain_t7_t12_wmc15_IT_bi_img1000': 99,
    'CORnet-S_brain_t7_t12_wmc15_IT_bi_img100': 81,
    'CORnet-S_brainboth2_IT_trconv3_bi_img500000': 23,
    'CORnet-S_brainboth2_IT_trconv3_bi_img100': 77,
    'CORnet-S_brainboth2_IT_trconv3_bi_img1000': 99,
    'CORnet-S_brainboth2_IT_trconv3_bi_img10000': 31,
    'CORnet-S_brainboth2_IT_trconv3_bi_img100000': 54,

}

conv_to_norm = {
    'V1.norm1': 'V1.conv1',
    'V1.norm2': 'V1.conv2',
    'V2.norm_skip': 'V2.skip',
    'V2.norm1_0': 'V2.conv1',
    'V2.norm2_0': 'V2.conv2',
    'V2.norm3_0': 'V2.conv3',
    'V2.norm1_1': 'V2.conv1',
    'V2.norm2_1': 'V2.conv2',
    'V2.norm3_1': 'V2.conv3',
    'V4.norm_skip': 'V4.skip',
    'V4.norm1_0': 'V4.conv1',
    'V4.norm2_0': 'V4.conv2',
    'V4.norm3_0': 'V4.conv3',
    'V4.norm1_1': 'V4.conv1',
    'V4.norm2_1': 'V4.conv2',
    'V4.norm3_1': 'V4.conv3',
    'V4.norm1_2': 'V4.conv1',
    'V4.norm2_2': 'V4.conv2',
    'V4.norm3_2': 'V4.conv3',
    'V4.norm1_3': 'V4.conv1',
    'V4.norm2_3': 'V4.conv2',
    'V4.norm3_3': 'V4.conv3',
    'IT.norm_skip': 'IT.skip',
    'IT.norm1_0': 'IT.conv1',
    'IT.norm2_0': 'IT.conv2',
    'IT.norm3_0': 'IT.conv3',
    'IT.norm1_1': 'IT.conv1',
    'IT.norm2_1': 'IT.conv2',
    'IT.norm3_1': 'IT.conv3',

}

layers = ['V1.conv1', 'V1.conv2',
          'V2.conv_input', 'V2.skip', 'V2.conv1', 'V2.conv2', 'V2.conv3',
          'V4.conv_input', 'V4.skip', 'V4.conv1', 'V4.conv2', 'V4.conv3',
          'IT.conv_input', 'IT.skip', 'IT.conv1', 'IT.conv2', 'IT.conv3']

# conv_to_norm = {
#     'module.V1.norm1.running_mean': 'V1.conv1',
#     'module.V1.norm1.running_var': 'V1.conv1',
#     'module.V1.norm2.running_mean': 'V1.conv2',
#     'module.V1.norm2.running_var': 'V1.conv2',
#     'module.V2.norm_skip.running_mean': 'V2.skip',
#     'module.V2.norm_skip.running_var': 'V2.skip',
#     'module.V2.norm1_0.running_mean': 'V2.conv1',
#     'module.V2.norm1_0.running_var': 'V2.conv1',
#     'module.V2.norm2_0.running_mean': 'V2.conv2',
#     'module.V2.norm2_0.running_var': 'V2.conv2',
#     'module.V2.norm3_0.running_mean': 'V2.conv3',
#     'module.V2.norm3_0.running_var': 'V2.conv3',
#     'module.V2.norm1_1.running_mean': 'V2.conv1',
#     'module.V2.norm1_1.running_var': 'V2.conv1',
#     'module.V2.norm2_1.running_mean': 'V2.conv2',
#     'module.V2.norm2_1.running_var': 'V2.conv2',
#     'module.V2.norm3_1.running_mean': 'V2.conv3',
#     'module.V2.norm3_1.running_var': 'V2.conv3',
#     'module.V4.norm_skip.running_mean': 'V4.skip',
#     'module.V4.norm_skip.running_var': 'V4.skip',
#     'module.V4.norm1_0.running_mean': 'V4.conv1',
#     'module.V4.norm1_0.running_var': 'V4.conv1',
#     'module.V4.norm2_0.running_mean': 'V4.conv2',
#     'module.V4.norm2_0.running_var': 'V4.conv2',
#     'module.V4.norm3_0.running_mean': 'V4.conv3',
#     'module.V4.norm3_0.running_var': 'V4.conv3',
#     'module.V4.norm1_1.running_mean': 'V4.conv1',
#     'module.V4.norm1_1.running_var': 'V4.conv1',
#     'module.V4.norm2_1.running_mean': 'V4.conv2',
#     'module.V4.norm2_1.running_var': 'V4.conv2',
#     'module.V4.norm3_1.running_mean': 'V4.conv3',
#     'module.V4.norm3_1.running_var': 'V4.conv3',
#     'module.V4.norm1_2.running_mean': 'V4.conv1',
#     'module.V4.norm1_2.running_var': 'V4.conv1',
#     'module.V4.norm2_2.running_mean': 'V4.conv2',
#     'module.V4.norm2_2.running_var': 'V4.conv2',
#     'module.V4.norm3_2.running_mean': 'V4.conv3',
#     'module.V4.norm3_2.running_var': 'V4.conv3',
#     'module.V4.norm1_3.running_mean': 'V4.conv1',
#     'module.V4.norm1_3.running_var': 'V4.conv1',
#     'module.V4.norm2_3.running_mean': 'V4.conv2',
#     'module.V4.norm2_3.running_var': 'V4.conv2',
#     'module.V4.norm3_3.running_mean': 'V4.conv3',
#     'module.V4.norm3_3.running_var': 'V4.conv3',
#     'module.IT.norm_skip.running_mean': 'IT.skip',
#     'module.IT.norm_skip.running_var': 'IT.skip',
#     'module.IT.norm1_0.running_mean': 'IT.conv1',
#     'module.IT.norm1_0.running_var': 'IT.conv1',
#     'module.IT.norm2_0.running_mean': 'IT.conv2',
#     'module.IT.norm2_0.running_var': 'IT.conv2',
#     'module.IT.norm3_0.running_mean': 'IT.conv3',
#     'module.IT.norm3_0.running_var': 'IT.conv3',
#     'module.IT.norm1_1.running_mean': 'IT.conv1',
#     'module.IT.norm1_1.running_var': 'IT.conv1',
#     'module.IT.norm2_1.running_mean': 'IT.conv2',
#     'module.IT.norm2_1.running_var': 'IT.conv2',
#     'module.IT.norm3_1.running_mean': 'IT.conv3',
#     'module.IT.norm3_1.running_var': 'IT.conv3',
#
# }
