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

base_dir = '/home/franzi/Projects/weight_initialization'
# dir = '/braintree/home/fgeiger/weight_initialization'

best_models_brain_avg_all = {
    # 'CORnet-S_full': 'Full',
    # 'CORnet-S_brain3_kn8_kn9_kn10_kn11_kn12_tra_bi' : 'V4.conv3_special',
    # 'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_tr_bi' : 'V4.conv3_special',
    # 'CORnet-S_brain3_knall_IT_bi': 'IT.conv3',
    # 'CORnet-S_brain3_t7_t12_knall_IT_bi' : 'IT.conv3_special', old
    # 'CORnet-S_brain_t7_t12_knk15_IT_bi' : 'IT.conv3_special',
    # 'CORnet-S_train_gmk1_bd2_bi': 'V1.conv2',
    # 'CORnet-S_train_gmk1_gmk2_ln3_bi': 'V2.input',
    # 'CORnet-S_train_gmk1_gmk2_kn3_ln4_bi': 'V2.skip',
    # 'CORnet-S_train_gmk1_gmk2_ln3_kn4_ln5_bi': 'V2.conv1',
    # 'CORnet-S_train_gmk1_gmk2_kn3_mi4_kn5_bd6_bi': 'V2.conv2',
    # 'CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wm6_bi': 'V2.conv3_special',
    'CORnet-S_train_gmk1_cl2_7_bi_seed42': 'V2.conv3',
    'CORnet-S_cluster2_v2_V4_bi_seed42': 'V4.conv3',
    # 'CORnet-S_brain_wmc15_IT_bi': 'IT.conv3',
    'CORnet-S_cluster2_v2_IT_bi_seed42': 'IT.conv3',
}
best_special_brain = {
    'CORnet-S_brain_t7_t12_wmc15_IT_bi': 'IT.conv3_special',
    'CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_tr_bi': 'V4.conv3_special',
    'CORnet-S_train_gmk1_gmk2_kn3_mi4_kn5_bd6_bi': 'V2.conv3_special',
}
best_special_brain_2 = {
    # 'CORnet-S_cluster2_v2_IT_trconv3_bi_seed42': 'IT.conv3_special',
    'CORnet-S_cluster2_v2_IT_trconv3_bi': 'IT.conv3_special',
    'CORnet-S_cluster2_v2_V4_trconv3_bi': 'V4.conv3_special',
    'CORnet-S_train_gmk1_cl2_7_7tr_bi': 'V2.conv3_special',
}

no_init_conv3_train = {
    'CORnet-S_train_conv3_bi': 'IT.conv3_special',
    'CORnet-S_train_conv3_V4_bi': 'V4.conv3_special',
    'CORnet-S_train_conv3_V2_bi': 'V2.conv3_special'
}

no_gabor_conv3_train = {
    'CORnet-S_cluster9_IT_trconv3_bi': 'IT.conv3_special',
    'CORnet-S_cluster9_V4_trconv3_bi': 'V4.conv3_special',
    'CORnet-S_train_wmk1_cl2_7_7tr_bi': 'V2.conv3_special'
}

no_gabor_no_conv3_train = {
    'CORnet-S_brain2_knall_IT_bi_v2': 'IT.conv3',
    'CORnet-S_brain2_knall_V4_bi_v2': 'V4.conv3',
    'CORnet-S_train_kn1_kn2_kn3_kn4_kn5_kn6_kn7_bi_v2': 'V2.conv3'
}

all_cluster = {
    'CORnet-S_cluster10_V4_trconv3_bi': 'V4.conv3_special',
    'CORnet-S_cluster10_IT_trconv3_bi': 'IT.conv3_special',

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
    'V2.conv3_special': 12,
    'mobilenet': 27
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
    # 'CORnet-S_train_V2': 16,
    'CORnet-S_train_V4': 12,
    'CORnet-S_train_IT_seed_0': 6,
    'CORnet-S_train_random': 1,
    # 'CORnet-S_random': 0
}

layer_random_small = {
    # 'CORnet-S_full': 17,
    # 'CORnet-S_train_V2': 16,
    # 'CORnet-S_train_V4': 12,
    # 'CORnet-S_train_IT_seed_0': 6,
    'CORnet-S_train_random': 1,
    # 'CORnet-S_random': 0
}

random_scores = {
    'CORnet-S_full': 'Standard training',
    # 'CORnet-S_train_V2': 'V1.conv2',
    'CORnet-S_train_V4': 'V2.conv3',
    'CORnet-S_train_IT_seed_0': 'V4.conv3',
    'CORnet-S_train_random': 'IT.conv3',
    # "CORnet-S_random": 'decoder',
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
    'CORnet-S_full_seed42': 29,
    'CORnet-S_full_seed94': 28,
    'CORnet-S_train_V2': 20,
    'CORnet-S_train_V2_seed42': 32,
    'CORnet-S_train_V4': 43,
    'CORnet-S_train_V4_seed42': 31,
    'CORnet-S_train_IT_seed_0': 37,
    'CORnet-S_train_IT_seed_0_seed42': 40,
    'CORnet-S_train_random': 28,
    'CORnet-S_train_random_seed42': 22,
    'CORnet-S_random': 0,
    'CORnet-S_brain_t7_t12_wmc15_IT_bi': 38,
    'resnet_v1_CORnet-S_full': 32,
    'resnet_v1_CORnet-S_train_random': 24,
    'resnet_v1_CORnet-S_brain_t7_t12_wmc15_bi': 26,
    'resnet_v1_CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wmc6_kn7_tr_bi': 24,
    'resnet_v1_CORnet-S_cluster2_v2_IT_trconv3_bi': 28,
    'resnet_v1_CORnet-S_cluster2_v2_IT': 28,
    'resnet_v3_CORnet-S_cluster2_v2_IT_trconv3_bi': 22,
    'resnet_v3_CORnet-S_cluster2_v2_V4_trconv3_bi': 28,
    'resnet_v3_CORnet-S_train_gmk1_cl2_7_7tr_bi': 32,
    'alexnet_v1_CORnet-S_brain_t7_t12_wmc15_IT_bi': 33,
    'alexnet_v1_CORnet-S_train_random': 29,
    'alexnet_v1_CORnet-S_full': 36,
    'alexnet_v1_CORnet-S_brain_kn8_kn9_kn10_wmc11_kn12_tr_bi': 33,
    'alexnet_v1_CORnet-S_train_gmk1_gmk2_kn3_kn4_kn5_wmc6_kn7_tr_bi': 46,
    'alexnet_v4_CORnet-S_train_gmk1_cl2_7_7tr_bi': 25,  # check
    'alexnet_v4_CORnet-S_cluster2_v2_V4_trconv3_bi': 32,  # check
    'alexnet_v4_CORnet-S_cluster2_v2_IT_trconv3_bi': 28,  # check
    'CORnet-S_cluster2_v2_V4_bi_seed42': 30,  # double check
    'CORnet-S_cluster2_v2_IT_bi_seed42': 31,
    'CORnet-S_cluster2_v2_IT_bi': 24,
    'CORnet-S_cluster2_v2_V4_trconv3_bi': 41,  # 27
    'CORnet-S_cluster2_v2_V4_trconv3_bi_seed42': 27,  # 27
    'CORnet-S_cluster2_v2_V4_trconv3_bi_seed94': 25,  # 27
    'CORnet-S_cluster2_v2_IT_trconv3_bi_seed42': 40,  # 40
    'CORnet-S_cluster2_v2_IT_trconv3_bi_seed94': 28,
    'CORnet-S_cluster2_v2_IT_trconv3_bi': 38,
    'CORnet-S_train_gmk1_cl2_7_7tr_bi': 33,  # 23
    'CORnet-S_train_gmk1_cl2_7_7tr_bi_seed42': 28,  # 23
    'CORnet-S_train_gmk1_cl2_7_bi_seed42': 34,
    'CORnet-S_train_conv3_bi': 29,  # 29 needs rework
    'CORnet-S_train_conv3_bi_seed42': 31,  # 29 needs rework
    'CORnet-S_train_conv3_V2_bi': 26,  # 25 needs rework
    'CORnet-S_train_conv3_V2_bi_seed42': 31,  # 25 needs rework
    'CORnet-S_train_conv3_V4_bi': 26,  # 26needs rework
    'CORnet-S_train_conv3_V4_bi_seed42': 37,  # 26needs rework
    'CORnet-S_cluster9_V4_trconv3_bi': 34,
    'CORnet-S_cluster9_IT_trconv3_bi': 44,
    'CORnet-S_train_wmk1_cl2_7_7tr_bi': 37,
    'mobilenet_v1_1.0_224': 100,
    'mobilenet_random': 0,
    'hmax': 0,
    'mobilenet_v1_CORnet-S_cluster2_v2_IT_trconv3_bi': 43,
    'mobilenet_v1_CORnet-S_cluster2_v2_V4_trconv3_bi': 36,
    'mobilenet_v1_CORnet-S_train_gmk1_cl2_7_7tr_bi': 34,
    'mobilenet_v2_CORnet-S_cluster2_v2_IT_trconv3_bi': 26,
    'mobilenet_v3_CORnet-S_cluster2_v2_IT_trconv3_bi': 28,
    'mobilenet_v4_CORnet-S_cluster2_v2_IT_trconv3_bi': 40,
    'mobilenet_v5_CORnet-S_cluster2_v2_IT_trconv3_bi': 29,
    'mobilenet_v6_CORnet-S_cluster2_v2_IT_trconv3_bi': 34,
    'mobilenet_v7_CORnet-S_cluster2_v2_IT_trconv3_bi': 43,
}
convergence_seed_42 = {
    'CORnet-S_train_V2_seed42': 32,
    'CORnet-S_full_seed42': 29,
    'CORnet-S_train_V4_seed42': 31,
    'CORnet-S_train_IT_seed_0': 40
}

convergence_images = {
    'CORnet-S_full_img500000': 21,
    'CORnet-S_full_img50000': 59,
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
    'CORnet-S_brain_t7_t12_wmc15_IT_bi_img1000_lr0.05': 88,
    'CORnet-S_cluster2_v2_IT_trconv3_bi_img500000': 26,
    'CORnet-S_cluster2_v2_IT_trconv3_bi_img50000': 41,
    'CORnet-S_cluster2_v2_IT_trconv3_bi_img100': 75,
    'CORnet-S_cluster2_v2_IT_trconv3_bi_img1000': 92,
    'CORnet-S_cluster2_v2_IT_trconv3_bi_img10000': 32,
    'CORnet-S_cluster2_v2_IT_trconv3_bi_img100000': 44,
    'CORnet-S_train_gmk1_cl2_7_7tr_bi_img100': 45,
    'CORnet-S_train_gmk1_cl2_7_7tr_bi_img1000': 80,
    'CORnet-S_train_gmk1_cl2_7_7tr_bi_img10000': 38,
    'CORnet-S_train_gmk1_cl2_7_7tr_bi_img100000': 72,
    'CORnet-S_train_gmk1_cl2_7_7tr_bi_img500000': 25,
    'CORnet-S_train_gmk1_cl2_7_7tr_bi_img50000': 43,
    'CORnet-S_cluster2_v2_V4_trconv3_bi_img100': 62,
    'CORnet-S_cluster2_v2_V4_trconv3_bi_img10000': 33,
    'CORnet-S_cluster2_v2_V4_trconv3_bi_img1000': 78,
    'CORnet-S_cluster2_v2_V4_trconv3_bi_img100000': 47,
    'CORnet-S_cluster2_v2_V4_trconv3_bi_img500000': 35,
    'CORnet-S_cluster2_v2_V4_trconv3_bi_img50000': 55,
    'CORnet-S_train_IT_seed_0_img100': 57,
    'CORnet-S_train_IT_seed_0_img1000': 70,
    'CORnet-S_train_IT_seed_0_img10000': 49,
    'CORnet-S_train_IT_seed_0_img100000': 55,
    'CORnet-S_train_IT_seed_0_img500000': 34,
    'CORnet-S_train_IT_seed_0_img50000': 53,
    'CORnet-S_train_V4_img100': 47,
    'CORnet-S_train_V4_img1000': 74,
    'CORnet-S_train_V4_img10000': 45,
    'CORnet-S_train_V4_img100000': 50,
    'CORnet-S_train_V4_img500000': 34,
    'CORnet-S_train_V4_img50000': 37,
    'CORnet-S_train_random_img100': 69,
    'CORnet-S_train_random_img1000': 57,
    'CORnet-S_train_random_img10000': 26,
    'CORnet-S_train_random_img100000': 35,
    'CORnet-S_train_random_img500000': 22,
    'CORnet-S_train_random_img50000': 28,
    'CORnet-S_train_conv3_bi_img100': 93,
    'CORnet-S_train_conv3_bi_img10000': 27,
    'CORnet-S_train_conv3_bi_img1000': 99,
    'CORnet-S_train_conv3_bi_img100000': 31,
    'CORnet-S_train_conv3_bi_img500000': 32,
    'CORnet-S_train_conv3_bi_img50000': 61,
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
    'V1.conv1': 'V1.conv1',
    'V1.conv2': 'V1.conv2',
    'V2.skip': 'V2.skip',
    'V2.conv1': 'V2.conv1',
    'V2.conv2': 'V2.conv2',
    'V2.conv3': 'V2.conv3',
    'V4.skip': 'V4.skip',
    'V4.conv1': 'V4.conv1',
    'V4.conv2': 'V4.conv2',
    'V4.conv3': 'V4.conv3',
    'IT.skip': 'IT.skip',
    'IT.conv1': 'IT.conv1',
    'IT.conv2': 'IT.conv2',
    'IT.conv3': 'IT.conv3',
    'V2.conv_input': 'V2.conv_input',
    'V4.conv_input': 'V4.conv_input',
    'IT.conv_input': 'IT.conv_input',
}

layers = ['V1.conv1', 'V1.conv2',
          'V2.conv_input', 'V2.skip', 'V2.conv1', 'V2.conv2', 'V2.conv3',
          'V4.conv_input', 'V4.skip', 'V4.conv1', 'V4.conv2', 'V4.conv3',
          'IT.conv_input', 'IT.skip', 'IT.conv1', 'IT.conv2', 'IT.conv3']