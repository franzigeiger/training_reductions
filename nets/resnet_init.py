from nets import get_config
from nets.test_models import get_resnet50

mapping = {'resnet.sth': 'V1.sth',
           'conv1.weight': 'V1.conv1',
           'bn1.weight': 'V1.norm1.weight',
           'bn1.bias': 'V1.norm1.bias',
           'layer1.0.conv1.weight': 'V1.conv1',
           'layer1.0.bn1.weight': 'V1.norm1_0.weight',
           'layer1.0.bn1.bias': 'V1.norm1_0.bias',
           'layer1.0.conv2.weight': 'V1.conv2',
           'layer1.0.bn2.weight': 'V1.norm2_0.weight',
           'layer1.0.bn2.bias': 'V1.norm2_0.bias',
           'layer1.0.conv3.weight': 'V1.conv3',
           'layer1.0.bn3.weight': 'V1.norm3_0.weight',
           'layer1.0.bn3.bias': 'V1.norm3_0.bias',
           'layer1.0.downsample.0.weight': 'V1.skip',
           'layer1.0.downsample.1.weight': 'V1.norm_skip.weight',
           'layer1.0.downsample.1.bias': 'V1.norm_skip.bias',
           'layer1.1.conv1.weight': 'V1.conv1',
           'layer1.1.bn1.weight': 'V1.norm1_1.weight',
           'layer1.1.bn1.bias': 'V1.norm1_1.bias',
           'layer1.1.conv2.weight': 'V1.conv2',
           'layer1.1.bn2.weight': 'V1.norm2_1.weight',
           'layer1.1.bn2.bias': 'V1.norm2_1.bias',
           'layer1.1.conv3.weight': 'V1.conv3',
           'layer1.1.bn3.weight': 'V1.norm3_1.weight',
           'layer1.1.bn3.bias': 'V1.norm3_1.bias',
           'layer1.2.conv1.weight': 'V1.conv1',
           'layer1.2.bn1.weight': 'V1.norm1_0.weight',
           'layer1.2.bn1.bias': 'V1.norm1_0.bias',
           'layer1.2.conv2.weight': 'V1.conv2',
           'layer1.2.bn2.weight': 'V1.norm2_0.weight',
           'layer1.2.bn2.bias': 'V1.norm2_0.bias',
           'layer1.2.conv3.weight': 'V1.conv3',
           'layer1.2.bn3.weight': 'V1.norm3_0.weight',
           'layer1.2.bn3.bias': 'V1.norm3_0.bias',
           'layer2.0.conv1.weight': '',
           'layer2.0.bn1.weight': '',
           'layer2.0.bn1.bias': '',
           'layer2.0.conv2.weight': '',
           'layer2.0.bn2.weight': '',
           'layer2.0.bn2.bias': '',
           'layer2.0.conv3.weight': '',
           'layer2.0.bn3.weight': '',
           'layer2.0.bn3.bias': '',
           'layer2.0.downsample.0.weight': '',
           'layer2.0.downsample.1.weight': '',
           'layer2.0.downsample.1.bias': '',
           'layer2.1.conv1.weight': '',
           'layer2.1.bn1.weight': '',
           'layer2.1.bn1.bias': '',
           'layer2.1.conv2.weight': '',
           'layer2.1.bn2.weight': '',
           'layer2.1.bn2.bias': '',
           'layer2.1.conv3.weight': '',
           'layer2.1.bn3.weight': '',
           'layer2.1.bn3.bias': '',
           'layer2.2.conv1.weight': '',
           'layer2.2.bn1.weight': '',
           'layer2.2.bn1.bias': '',
           'layer2.2.conv2.weight': '',
           'layer2.2.bn2.weight': '',
           'layer2.2.bn2.bias': '',
           'layer2.2.conv3.weight': '',
           'layer2.2.bn3.weight': '',
           'layer2.2.bn3.bias': '',
           'layer2.3.conv1.weight': '',
           'layer2.3.bn1.weight': '',
           'layer2.3.bn1.bias': '',
           'layer2.3.conv2.weight': '',
           'layer2.3.bn2.weight': '',
           'layer2.3.bn2.bias': '',
           'layer2.3.conv3.weight': '',
           'layer2.3.bn3.weight': '',
           'layer2.3.bn3.bias': '',
           'layer3.0.conv1.weight': '',
           'layer3.0.bn1.weight': '',
           'layer3.0.bn1.bias': '',
           'layer3.0.conv2.weight': '',
           'layer3.0.bn2.weight': '',
           'layer3.0.bn2.bias': '',
           'layer3.0.conv3.weight': '',
           'layer3.0.bn3.weight': '',
           'layer3.0.bn3.bias': '',
           'layer3.0.downsample.0.weight': '',
           'layer3.0.downsample.1.weight': '',
           'layer3.0.downsample.1.bias': '',
           'layer3.1.conv1.weight': '',
           'layer3.1.bn1.weight': '',
           'layer3.1.bn1.bias': '',
           'layer3.1.conv2.weight': '',
           'layer3.1.bn2.weight': '',
           'layer3.1.bn2.bias': '',
           'layer3.1.conv3.weight': '',
           'layer3.1.bn3.weight': '',
           'layer3.1.bn3.bias': '',
           'layer3.2.conv1.weight': '',
           'layer3.2.bn1.weight': '',
           'layer3.2.bn1.bias': '',
           'layer3.2.conv2.weight': '',
           'layer3.2.bn2.weight': '',
           'layer3.2.bn2.bias': '', }


# 'layer3.2.conv3.weight'
# 'layer3.2.bn3.weight'
# 'layer3.2.bn3.bias'
# 'layer3.3.conv1.weight'
# 'layer3.3.bn1.weight'
# 'layer3.3.bn1.bias'
# 'layer3.3.conv2.weight'
# 'layer3.3.bn2.weight'
# 'layer3.3.bn2.bias'
# 'layer3.3.conv3.weight'
# 'layer3.3.bn3.weight'
# 'layer3.3.bn3.bias'
# 'layer3.4.conv1.weight'
# 'layer3.4.bn1.weight'
# 'layer3.4.bn1.bias'
# 'layer3.4.conv2.weight'
# 'layer3.4.bn2.weight'
# 'layer3.4.bn2.bias'
# 'layer3.4.conv3.weight'
# 'layer3.4.bn3.weight'
# 'layer3.4.bn3.bias'
# 'layer3.5.conv1.weight'
# 'layer3.5.bn1.weight'
# 'layer3.5.bn1.bias'
# 'layer3.5.conv2.weight'
# 'layer3.5.bn2.weight'
# 'layer3.5.bn2.bias'
# 'layer3.5.conv3.weight'
# 'layer3.5.bn3.weight'
# 'layer3.5.bn3.bias'
# 'layer4.0.conv1.weight'
# 'layer4.0.bn1.weight'
# 'layer4.0.bn1.bias'
# 'layer4.0.conv2.weight'
# 'layer4.0.bn2.weight'
# 'layer4.0.bn2.bias'
# 'layer4.0.conv3.weight'
# 'layer4.0.bn3.weight'
# 'layer4.0.bn3.bias'
# 'layer4.0.downsample.0.weight'
# 'layer4.0.downsample.1.weight'
# 'layer4.0.downsample.1.bias'
# 'layer4.1.conv1.weight'
# 'layer4.1.bn1.weight'
# 'layer4.1.bn1.bias'
# 'layer4.1.conv2.weight'
# 'layer4.1.bn2.weight'
# 'layer4.1.bn2.bias'
# 'layer4.1.conv3.weight'
# 'layer4.1.bn3.weight'
# 'layer4.1.bn3.bias'
# 'layer4.2.conv1.weight'
# 'layer4.2.bn1.weight'
# 'layer4.2.bn1.bias'
# 'layer4.2.conv2.weight'
# 'layer4.2.bn2.weight'
# 'layer4.2.bn2.bias'
# 'layer4.2.conv3.weight'
# 'layer4.2.bn3.weight'
# 'layer4.2.bn3.bias'
# [3,4,6,3] [2,4,2]
# ['conv1'] +
# [''layer1.0.conv3', ''layer1.1.conv3', ''layer1.2.conv3'] +
# [''layer2.0.downsample.0', ''layer2.1.conv3', ''layer2.2.conv3', ''layer2.3.conv3'] +
# [''layer3.0.downsample.0', ''layer3.1.conv3', ''layer3.2.conv3', ''layer3.3.conv3',
#  ''layer3.4.conv3', ''layer3.5.conv3'] +
# [''layer4.0.downsample.0', ''layer4.1.conv3', ''layer4.2.conv3'] +
# ['avgpool']

def train_resnet(identifier='resnet50', train_func=None):
    config = get_config('CORnet-S_train_gmk1_gmk2_ln3_ln4_ln5_wm6_ra')
    model = get_resnet50(False)
    resnet_config = {}

    # for name, m in model.named_parameters():
    #     print(name)

    for layer, cornet in mapping.items():
        if not any(module in layer for module in config['layers']):
            if cornet in config:
                resnet_config[layer] = config[cornet]
            if 'bn' in layer:
                resnet_config[layer] = cornet

    # call initer, train
    # run_model_training(model=model,config=config, train_func=train_func)
