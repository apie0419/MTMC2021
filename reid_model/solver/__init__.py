from .optim import make_optimizer
from .lr_scheduler import WarmupMultiStepLR
import torch



"""def init_optim(optim_name, model, has_center=False, center_criterion=None):
    if optim_name == 'Adam' and has_center==True:
        optimizer, optimizer_center = optimizer_with_center(model, center_criterion)
        return optimizer, optimizer_center

def init_scheduler(sched_name, optimizer, milestones, gamma):
    if sched_name == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)"""