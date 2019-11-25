
import torch
import torch.nn as nn

def apply_to_net(net, config):
    def init_weights(m):
        if type(m) == nn.Conv2d or ((type(m) == nn.Linear or type(m) == nn.BatchNorm2d) and batchnorm_shuffle):
            function(m)

    net.apply(init_weights)
    return net