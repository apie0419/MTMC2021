import os, re, torch, cv2, math
import numpy as np
import torch.multiprocessing as mp

from tqdm          import tqdm
from utils         import init_path, check_setting
from utils.mct     import build_model
from utils.utils   import getdistance, cosine
from utils.objects import Track

init_path()

from config import cfg

check_setting(cfg)

ROOT_DIR    = cfg.PATH.ROOT_PATH
INPUT_DIR   = cfg.PATH.INPUT_PATH
NUM_WORKERS = mp.cpu_count()
DEVICE      = cfg.DEVICE.TYPE
GPUS        = cfg.DEVICE.GPUS
device      = torch.device(f"{DEVICE}:{GPUS[0]}")

write_lock  = mp.Lock()
count_id    = 0

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
        ts = float(l[6])
        gps = list(map(float, l[7:9]))
        features = np.array(list(map(float, l[9:])), np.float32)
        
        write_lock.acquire()
        if id not in data[camera_dir]:
            data[camera_dir][id] = Track()
        track = data[camera_dir][id]
        track.gps_list.append(gps)
        track.feature_list.append(features)
        track.frame_list.append(frame_id)
        track.box_list.append(box)
        track.ts_list.append(ts)
        track.id = id
        write_lock.release()

    return data

def get_scene_camera_dict(camera_dirs):
    res = dict()
    for camera_dir in camera_dirs:
        camera = camera_dir.split('/')[-1]
        scene = camera_dir.split('/')[-2]
        res[camera] = scene

    return res

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
        files.append(os.path.join(camera_dir, f"{cfg.SCT}_{cfg.DETECTION}_all_features_post.txt"))
        
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
            for track in results[camera]:
                obj_id = track.id
                for i in range(len(track)):
                    xmin, ymin, width, height = track.box_list[i]
                    xworld, yworld = track.gps_list[i]
                    frame_id = track.frame_list[i]
                    line = str(camera_id) + ',' + str(obj_id) + ',' + str(frame_id) + ',' + \
                            str(xmin) + ',' + str(ymin) + ',' + str(width) + ',' + str(height) + ',' + \
                            str(xworld) + ',' + str(yworld) + "\n"
                    f.write(line)

def match_track_by_cosine(query_ft, gallery_fts):

    affinity_list = list()
    for gallery_ft in gallery_fts:
        aff = cosine(query_ft, gallery_ft)
        affinity_list.append(aff)
    
    ind = np.argmax(affinity_list)
    if affinity_list[ind] >= 0.7:
        match_idx = ind
    else:
        match_idx = -1

    return match_idx
        
def match_track(model, query_ft, gallery_fts):
    
    tracklets = [query_ft]
    tracklets.extend(gallery_fts)

    data = torch.stack(tracklets)
    with torch.no_grad():
        data = data.to(device)
        preds = model(data)
        if preds.max().item() > 0.6:
            match_idx = preds.argmax().item()
        else:
            match_idx = -1
    
    return match_idx

def main(data, camera_dirs):
    global count_id
    results  = dict()
    first_camera = camera_dirs[0].split("/")[-1]
    results[first_camera] = list(data[first_camera].values())
    count_id = len(results[first_camera]) + 1
    scene_dict = get_scene_camera_dict(camera_dirs)
    model = build_model(cfg, device)
    model.eval()

    for camera_dir in camera_dirs[1:]:
        camera = camera_dir.split("/")[-1]
        query_scene = scene_dict[camera]
        query_tracks = data[camera]
        results[camera] = list()
        match_ids = list()
        

        for obj_id in tqdm(query_tracks, desc=f"Processing Camera Dir {camera}"):
            gallery_fts = list()
            query_track = query_tracks[obj_id]
            speed = query_track.speed()
            query_ft = torch.tensor(query_track.feature_list)
            mean = query_ft.mean(dim=0)
            std = query_ft.std(dim=0, unbiased=False)
            query_ft = torch.cat((mean, std))
            
            gids = list()
            for c in results:
                gallery_scene = scene_dict[c]
                if gallery_scene != query_scene:
                    continue
                for gallery_track in results[c]:
                    gid = gallery_track.id
                    
                    g_x = gallery_track.gps_list[int(len(gallery_track)/2)][0] - gallery_track.gps_list[0][0]
                    g_y = gallery_track.gps_list[int(len(gallery_track)/2)][1] - gallery_track.gps_list[0][1]
                    q_x = query_track.gps_list[int(len(query_track)/2)][0] - query_track.gps_list[0][0]
                    q_y = query_track.gps_list[int(len(query_track)/2)][1] - query_track.gps_list[0][1]
                    vec1 = [g_x, g_y]
                    vec2 = [q_x, q_y]
                    direction = cosine(vec1, vec2)
                    dis_ts = abs(gallery_track.ts_list[0] - query_track.ts_list[0])
                    if np.isnan(direction):
                        continue
                    expected_time = getdistance(query_track.gps_list[0], gallery_track.gps_list[0]) / speed
                    
                    if (abs(dis_ts - expected_time) > 10) or (direction < 0.5) or (gid in match_ids):
                        continue
                    
                    gallery_ft = torch.tensor(gallery_track.feature_list)
                    mean = gallery_ft.mean(dim=0)
                    std  = gallery_ft.std(dim=0, unbiased=False)
                    gallery_ft = torch.cat((mean, std))
                    if gid in gids:
                        old_gft = gallery_fts[gids.index(gid)]
                        if cosine(gallery_ft, query_ft) > cosine(old_gft, query_ft):
                            gallery_fts[gids.index(gid)] = gallery_ft
                    else:
                        gids.append(gallery_track.id)
                        gallery_fts.append(gallery_ft)

            match = False
            if len(gallery_fts) != 0:
                match_idx = match_track(model, query_ft, gallery_fts)
                # match_idx = match_track_by_cosine(query_ft, gallery_fts)
                if match_idx != -1:
                    match_id = gids[match_idx]
                    match = True

                    query_track.id = match_id
                    match_ids.append(match_id)

            if not match:
                query_track.id = count_id
                count_id += 1

            results[camera].append(query_track)
        
    write_results(results, camera_dirs)
    
if __name__ == "__main__":
    data, camera_dirs = prepare_data()
    main(data, camera_dirs)
    