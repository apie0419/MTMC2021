from options import opt
from pathlib import Path
import torch


def make_optimizer(model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = opt.base_lr
        weight_decay = opt.weight_decay
        if "bias" in key:
            lr = opt.base_lr * opt.bias_lr_factor
            weight_decay = opt.weight_decay_bias
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if opt.optim_name == 'SGD':
        optimizer = getattr(torch.optim, opt.optim_name)(params, momentum=opt.momentum)
    else:
        optimizer = getattr(torch.optim, opt.optim_name)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=opt.center_lr)

    return optimizer, optimizer_center


"""def optimizer_with_center(model, center_criterion):
    normal_lr = opt.base_lr
    center_lr = opt.center_lr
    weight_decay = opt.weight_decay

    optimizer = torch.optim.Adam(model.parameters(), lr=normal_lr, weight_decay=weight_decay, amsgrad=True)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=center_lr)

    return optimizer, optimizer_center"""

