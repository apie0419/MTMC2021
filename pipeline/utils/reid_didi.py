import torch
import torchvision.transforms as T

from . import init_path

init_path()

from reid_model_didi.modeling import Baseline

def build_model(cfg):
    PRETRAIN_PATH = "/home/apie/projects/AIC20-MTMC/weights/resnet50-19c8e357.pth"
    WEIGHT        = cfg.REID.WEIGHTS
    weight = torch.load(WEIGHT)
    NUM_CLASSES = weight["classifier.weight"].shape[0]
    model = Baseline(NUM_CLASSES, 1, PRETRAIN_PATH)
    model.load_state_dict(weight)
    return model

def build_transform(cfg):
    transform=T.Compose([
        transforms.Resize(cfg.REID.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    ])
    return transform