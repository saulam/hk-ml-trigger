import torch


def weight_norm(net):
    """Compute the sum of squared L2 norms of all parameters in a network."""
    norm = 0
    for param in net.parameters():
        norm += (param ** 2).sum()
    return norm
