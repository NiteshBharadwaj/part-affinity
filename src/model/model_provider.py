from .vgg import VGG
from .paf_model import PAFModel
import torch.nn as nn
import torch


def parse_criterion(criterion):
    if criterion == 'l1':
        return nn.L1Loss()
    elif criterion == 'mse':
        return nn.MSELoss()


def create_model(opt):
    if opt.model == 'vgg':
        backend = VGG()
        backend_feats = 128
    else:
        raise ValueError('Model ' + opt.model + ' not available.')
    model = PAFModel(backend,backend_feats,18,32)
    criterion_hm = parse_criterion(opt.criterionHm)
    criterion_paf = parse_criterion(opt.criterionPaf)
    return model, criterion_hm, criterion_paf

def create_optimizer(opt, model):
    return torch.optim.Adam(model.parameters(), opt.LR)