from utils.utils import getdistance
import numpy as np, os
from utils import init_path, check_setting
init_path()
from config import cfg
check_setting(cfg)

INPUT_DIR   = cfg.PATH.INPUT_PATH

def read(scene, camera, _id):
    file = os.path.join(INPUT_DIR, scene, camera, "res_opt.txt")
    f = open(file, "r")
    data = dict()
    for line in f.readlines():
        l = line.strip("\n").split(",")
        frame_id = int(l[2])
        if _id == int(l[1]):
            gps = np.array(list(map(float, l[-2:])))
            data[frame_id] = gps
    return data

if __name__ == "__main__":
    # scene = "S05"
    # camera = "c027"
    # _id = 4
    # data = read(scene, camera, _id)
    # data = sorted(data.items(), key=lambda r: r[0])
    
    # pt1 = data[0][1]
    # for d in data[1:]:
    #     pt2 = d[1]
    #     print (getdistance(pt1, pt2))