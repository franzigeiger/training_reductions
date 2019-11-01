from numpy.random.mtrand import RandomState
from scipy.stats import norm


def apply_norm_dist(net):
    import torch
    import torch.nn as nn
    def init_weights(m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            print(type(m))
            weights = m.weight.data.cpu().numpy()
            mu, std = norm.fit(weights)
            torch.nn.init.normal(m.weight, mean=mu, std=std)

    net.apply(init_weights)
    return net


def apply_jumbler(net):
    import torch
    import torch.nn as nn
    def init_weights(m):
        if type(m) == nn.Conv2d:
            weights = m.weight.data.cpu().numpy()
            random_state = RandomState()
            random_order = random_state.permutation(weights.shape[1])
            weights = weights[:, random_order]
            m.weight.data = torch.Tensor(weights)

    net.apply(init_weights)
    return net


def apply_uniform_dist(net):
    import torch
    import torch.nn as nn
    def init_weights(m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            weights = m.weight.data.cpu().numpy()
            mu, std = norm.fit(weights)
            torch.nn.init.uniform(m.weight, a=-1 * std, b=std)

    net.apply(init_weights)
    return net
