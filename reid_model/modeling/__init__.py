from reid_model.options import opt
from pathlib import Path
from .baseline import Backbone
from torchvision import transforms
import torch


__all__ = ['resnet18',"resnet34","resnet50","resnet101","resnet152","senet154","se_resnet50","se_resnet101","se_resnet152","se_resnext50_32x4d","se_resnext101_32x4d","resnet50_ibn_a", "resnet101_ibn_a"]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50':  Path(opt.pretrained_dir).joinpath('resnet50-19c8e357.pth'),
    'resnet101': Path(opt.pretrained_dir).joinpath('resnet101-5d3b4d8f.pth'),
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'se_resnet50': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
    'se_resnet101': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
    'se_resnet152': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
    'se_resnext50': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
    'se_resnext101': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
    'senet154': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
    'resnet50_ibn_a': Path(opt.pretrained_dir).joinpath('r50_ibn_a.pth'),
    'resnet101_ibn_a': Path(opt.pretrained_dir).joinpath('resnet101_ibn_a-59ea0ac6.pth')
}

def make_model(num_classes,pretrain_path=str(model_urls[opt.reid_model])):
    model = Backbone(num_classes, pretrain_path)
    return model


# data transform for test
def build_transform(cfg):
    normalize_transform=transforms.Normalize(mean=cfg.REID.PIXEL_MEAN,std=cfg.REID.PIXEL_STD)
    transform=transforms.Compose([
        transforms.Resize(cfg.REID.SIZE_TEST),
        transforms.ToTensor(),
        normalize_transform
    ])
    return transform


# load reid model in test
def build_appearance_model(cfg):
    WEIGHT=cfg.MODEL.APPEARANCE.WEIGHTS
    NUM_CLASSES=cfg.MODEL.APPEARANCE.NUM_CLASSES
    model=make_model(NUM_CLASSES)
    checkpoint=torch.load(WEIGHT)
    model.load_state_dict(checkpoint['state_dict'])
    return model
    