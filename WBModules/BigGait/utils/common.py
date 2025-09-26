import copy

import torch
import numpy as np
import torch.autograd as autograd
from torch import nn


def list2var(x, **kwargs):
    return np2var(np.array(x), **kwargs)


def np2var(x, **kwargs):
    return ts2var(torch.from_numpy(x), **kwargs)


def ts2var(x, **kwargs):
    return autograd.Variable(x, **kwargs)



def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))


def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def get_attr_from(sources, name):
    try:
        return getattr(sources[0], name)
    except:
        return get_attr_from(sources[1:], name) if len(sources) > 1 else getattr(sources[0], name)


def is_list(x):
    return isinstance(x, list) or isinstance(x, nn.ModuleList)