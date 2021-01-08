import os, sys

def init_path():
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    sys.path.append("/".join(BASE_PATH.split("/")[:-2]))

def check_setting(cfg):
    _type = cfg.PATH.INPUT_PATH.split("/")[-1]
    if _type == "test":
        assert cfg.SCT == "tnt"
        assert cfg.DETECTION == "mask_rcnn"
    elif _type == "validation" or _type == "train":
        assert cfg.SCT == "tc" or cfg.SCT == "deepsort" or cfg.SCT == "moana"
        assert cfg.DETECTION == "mask_rcnn" or cfg.DETECTION == "yolo3" or cfg.DETECTION == "ssd512"