import os, cv2
import numpy as np

from tqdm  import tqdm
from utils import init_path, check_setting

init_path()

from config import cfg

check_setting(cfg)

INPUT_DIR = cfg.PATH.INPUT_PATH

SIDE_TH = 25

def get_video_size(video_path):
    cap = cv2.VideoCapture(video_path)
    w = cap.get(3)
    h = cap.get(4)
    video_size = (h, w)
    return video_size

def adaptation_box(box, resolution):
    h, w = resolution
    p0x = int(max(0, box[0] - SIDE_TH))
    p0y = int(max(0, box[1] - SIDE_TH))
    p1x = int(min(box[0] + box[2] + SIDE_TH, w-1))
    p1y = int(min(box[1] + box[3] + SIDE_TH, h-1))
    return [p0x, p0y, p1x-p0x, p1y-p0y]

def write_results(output_file, results, camera):
    with open(output_file, 'w') as f:
        for frame_index, data in results.items():
            for det in data:
                id_str = str(det["id"])
                box_str = ",".join(list(map(str, det["box"])))
                world_str = ",".join(list(map(str, det["world"])))
                camera_id_str = str(int(camera[1:]))
                line = camera_id_str + ',' + id_str + ',' + str(frame_index) + ',' + \
                    box_str + ',' + world_str + '\n'
                f.write(line)
        

def main(result_file, size):
    res = dict()
    with open(result_file, 'r') as f:
        for line in f.readlines():
            words = line.strip('\n').split(',')
            camera = words[0]
            id = words[1]
            frame_index = words[2]
            box = [int(words[3]), int(words[4]), int(words[5]), int(words[6])]
            world = [float(words[7]), float(words[8])]
            box = adaptation_box(box, size)
            if frame_index not in res:
                res[frame_index] = list()
            
            data = {
                "id": id,
                "box": box,
                "world": world
            }

            res[frame_index].append(data)

    return res

if __name__ == "__main__":
    for scene_dir in os.listdir(INPUT_DIR):
        if scene_dir.startswith("S0"):
            for camera_dir in os.listdir(os.path.join(INPUT_DIR, scene_dir)):
                if camera_dir.startswith("c0"):
                    video_path = os.path.join(INPUT_DIR, scene_dir, camera_dir, 'vdo.avi')
                    result_file = os.path.join(INPUT_DIR, scene_dir, camera_dir, "res.txt")
                    output_file = os.path.join(INPUT_DIR, scene_dir, camera_dir, "res_opt.txt")
                    video_size = get_video_size(video_path)
                    results = main(result_file, video_size)
                    write_results(output_file, results, camera_dir)

                    
                    
