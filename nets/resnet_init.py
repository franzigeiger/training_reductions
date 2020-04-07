from nets import get_config
from nets.test_models import get_resnet50

mapping = {'resnet.sth': 'V1.sth'}


# ['conv1'] +
# ['layer1.0.conv3', 'layer1.1.conv3', 'layer1.2.conv3'] +
# ['layer2.0.downsample.0', 'layer2.1.conv3', 'layer2.2.conv3', 'layer2.3.conv3'] +
# ['layer3.0.downsample.0', 'layer3.1.conv3', 'layer3.2.conv3', 'layer3.3.conv3',
#  'layer3.4.conv3', 'layer3.5.conv3'] +
# ['layer4.0.downsample.0', 'layer4.1.conv3', 'layer4.2.conv3'] +
# ['avgpool']

def train_resnet(identifier='resnet50', train_func=None):
    config = get_config('best_guess')
    model = get_resnet50(identifier, False, config)
    resnet_config = {}

    # for layer, cornet in mapping:
    #     if cornet in
    # run_model_training(model=model,config=config, train_func=train_func)
