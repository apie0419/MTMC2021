import os, tarfile

from tqdm  import tqdm
from utils import init_path, check_setting

init_path()

from config import cfg

check_setting(cfg)

INPUT_DIR = cfg.PATH.INPUT_PATH

def main():
    camera_dirs = list()
    for scene_dir in os.listdir(INPUT_DIR):
        if scene_dir.startswith("S0"):
            for camera_dir in os.listdir(os.path.join(INPUT_DIR, scene_dir)):
                if camera_dir.startswith("c0"):
                    camera_dirs.append(os.path.join(INPUT_DIR, scene_dir, camera_dir))
    
    results_filename = os.path.join(INPUT_DIR, "sct.txt")
    with open(results_filename, "w+") as f:
        for camera_dir in camera_dirs:
            tmp_result_file = os.path.join(camera_dir, f"{cfg.SCT}_{cfg.DETECTION}_all_features_post.txt")
            with open(tmp_result_file, "r") as tmp_f:
                for line in tmp_f.readlines():
                    words = line.split(',')
                    camera = camera_dir.split('/')[-1]
                    camera_id_str = str(int(camera[1:]))
                    frame_id = words[0]
                    obj_id = words[1]
                    box = ",".join(words[2:6])
                    world = ",".join(words[7:9])
                    l = camera_id_str + ',' + obj_id + ',' + frame_id + ',' + box + ',' + world + '\n'
                    f.write(l)

if __name__ == "__main__":
    main()