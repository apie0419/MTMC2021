from yacs.config import CfgNode as CN
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

_C = CN()
_C.PATH = CN()
_C.DEVICE = CN()
_C.REID = CN()
_C.MCT = CN()

_C.SCT = "tc"
_C.DETECTION = "mask_rcnn"

_C.PATH.ROOT_PATH = '<path_project_dir>'
_C.PATH.INPUT_PATH = '<path_to_input_path>' # train or validation or test

_C.DEVICE.GPUS = [1, 2, 3] # gpu number
_C.DEVICE.TYPE = "<cuda or cpu>"

_C.REID.WEIGHTS = "<path_to_reid_model_weight>"
_C.REID.IMG_SIZE = [224, 224]
_C.REID.NUM_CLASSES = 667
_C.REID.EMB_SIZE = 2048
_C.REID.BATCH_SIZE = 64

_C.MCT.FEATURE_DIM = 2048
_C.MCT.WEIGHT = '<path_to_weight>'
_C.MCT.METHOD = "CIR"
_C.MCT.METRIC = "cosine"
_C.MCT.SIM_TH = 0.8
_C.MCT.CIR_TH = 0.8
_C.MCT.RW = False
_C.merge_from_file(os.path.join(BASE_PATH, "config.yaml"))