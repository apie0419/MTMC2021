from yacs.config import CfgNode as CN
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

_C = CN()
_C.PATH = CN()
_C.DEVICE = CN()
_C.MCT = CN()

_C.PATH.INPUT_PATH = '<path_to_input_path>'

_C.DEVICE.GPUS = [1, 2, 3] # gpu number
_C.DEVICE.TYPE = "<cuda or cpu>"

_C.MCT.E_LAYERS = 3
_C.MCT.FEATURE_DIM = 2048
_C.MCT.LEARNING_RATE = 0.02

_C.merge_from_file(os.path.join(BASE_PATH, "config.yaml"))