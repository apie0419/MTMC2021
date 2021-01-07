import os, sys

def init_path():
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    sys.path.append("/".join(BASE_PATH.split("/")[:-2]))