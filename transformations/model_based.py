import torch
import torch.nn as nn

def apply_to_one_layer(net, function, config):
    apply_to_one_layer.layer = config[0]
    apply_to_one_layer.counter = 0

    def init_weights(m):
        if type(m) == nn.Conv2d or ((type(m) == nn.Linear or type(m) == nn.BatchNorm2d) and False):
            apply_to_one_layer.counter += 1
            if apply_to_one_layer.counter is apply_to_one_layer.layer:
                function(m)

    net.apply(init_weights)
    return net
