import os, re
import numpy as np
import multiprocessing as mp

from tqdm  import tqdm
from utils import init_path, check_setting

init_path()

from config import cfg

check_setting(cfg)

INPUT_DIR   = cfg.PATH.INPUT_PATH
NUM_WORKERS = mp.cpu_count()
DEVICE      = cfg.DEVICE.TYPE
GPUS        = cfg.DEVICE.GPUS
write_lock  = mp.Lock()


class Track(object):

    def __init__(self):
        self.gps_list = list()
        self.feature_list = list()
        self.frame_list = list()
        self.box_list = list()
    
    def __len__(self):
        return len(self.frame_list)

def read_features_file(file):
    camera_dir = file.split("/")[-2]
    f = open(file, "r")
    data = dict()
    data[camera_dir] = dict()

    for line in f.readlines():
        l = line.strip("\n").split(",")
        frame_id = int(l[0])
        id = int(l[1])
        box = list(map(int, list(map(float, l[2:6]))))
        gps = list(map(float, l[6:8]))
        features = np.array(list(map(float, l[8:])), np.float)
        
        write_lock.acquire()
        if id not in data[camera_dir]:
            data[camera_dir][id] = Track()
        track = data[camera_dir][id]
        track.gps_list.append(gps)
        track.feature_list.append(features)
        track.frame_list.append(frame_id)
        track.box_list.append(box)
        write_lock.release()

    return data

def prepare_data():
    data = dict()
    camera_dirs = list()
    for scene_dir in os.listdir(INPUT_DIR):
        if scene_dir.startswith("S0"):
            for camera_dir in os.listdir(os.path.join(INPUT_DIR, scene_dir)):
                if camera_dir.startswith("c0"):
                    camera_dirs.append(os.path.join(INPUT_DIR, scene_dir, camera_dir))
    
    files = list()
    for camera_dir in camera_dirs:
        files.append(os.path.join(camera_dir, "all_features.txt"))
        
    pool = mp.Pool(NUM_WORKERS)

    for d in tqdm(pool.imap_unordered(read_features_file, files), total=len(files), desc="Loading Data"):
        data.update(d)

    pool.close()
    pool.join()

    return data, camera_dirs

def write_results(results, camera_dirs):

    for camera_dir in tqdm(camera_dirs, desc="Writing Results"):
        result_file = os.path.join(camera_dir, "res.txt")
        camera = camera_dir.split("/")[-1]
        camera_id = int(re.search(r"([0-9]){3}", camera).group(0))
        with open(result_file, "w+") as f:
            for obj_id in results[camera]:
                track = results[camera][obj_id]
                for i in range(len(track)):
                    xmin, ymin, width, height = track.box_list[i]
                    xworld, yworld = track.gps_list[i]
                    frame_id = track.frame_list[i]
                    line = str(camera_id) + ',' + str(obj_id) + ',' + str(frame_id) + ',' + \
                            str(xmin) + ',' + str(ymin) + ',' + str(width) + ',' + str(height) + ',' + \
                            str(xworld) + ',' + str(yworld) + "\n"
                    f.write(line)
        

def match_track(query_tracks, gallery_tracks):
    results = list()
    """
    Doing Something
    """
    results = query_tracks # temp_result

    return results


def main(data, camera_dirs):
    results = dict()
    first_camera = camera_dirs[0].split("/")[-1]
    results[first_camera] = data[first_camera]
    for camera_dir in camera_dirs[1:]:
        camera = camera_dir.split("/")[-1]
        query_tracks = data[camera]
        gallery_tracks = list()
        for c in results:
            for track in results[c]:
                """
                filter some tracks
                """
                gallery_tracks.append(track)
        match_result = match_track(query_tracks, gallery_tracks)
        results[camera] = match_result


    write_results(results, camera_dirs)
    

if __name__ == "__main__":
    data, camera_dirs = prepare_data()

    main(data, camera_dirs)