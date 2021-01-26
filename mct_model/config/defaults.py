from yacs.config import CfgNode as CN
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

_C = CN()
_C.PATH = CN()
_C.DEVICE = CN()
_C.MCT = CN()

_C.PATH.INPUT_PATH = '<path_to_input_path>'
_C.PATH.VALID_PATH = '<path_to_valid_path>'
_C.PATH.OUTPUT_PATH = '<path_to_output_path>'

_C.DEVICE.GPU  = 4 # gpu number
_C.DEVICE.TYPE = "<cuda or cpu>"

_C.MCT.E_LAYERS = 3
_C.MCT.FEATURE_DIM = 2048
_C.MCT.LEARNING_RATE = 0.02
_C.MCT.EPOCHS = 10
_C.MCT.BATCH_SIZE = 64
_C.MCT.WEIGHT = '<path_to_weight>'

_C.merge_from_file(os.path.join(BASE_PATH, "config.yaml"))