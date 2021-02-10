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

write_lock  = mp.Lock()
read_lock   = mp.Lock()
stop        = mp.Value("i", False)
count_id    = mp.Value("i", 0)

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
        files.append(os.path.join(camera_dir, "all_features_post.txt"))
        
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

def match_track(device, data_queue, result):
    model = build_model(cfg, device)
    model.eval()
    
    while not data_queue.empty() or not stop.value:
        read_lock.acquire()
        try:
            query_track, gallery_tracks = data_queue.get(False)
        except:
            read_lock.release()
            continue
        read_lock.release()
        match = False
        query = torch.tensor(query_track.feature_list)
        mean = query.mean(dim=0)
        std = query.std(dim=0, unbiased=False)
        query = torch.cat((mean, std))
        tracklets = [query]
        ids = list()
        for track in gallery_tracks:
            gallery = torch.tensor(track.feature_list)
            mean = gallery.mean(dim=0)
            std  = gallery.std(dim=0, unbiased=False)
            gallery = torch.cat((mean, std))
            tracklets.append(gallery)
            ids.append(track.id)
            
        data = torch.stack(tracklets)
        with torch.no_grad():
            data = data.to(device)
            preds = model(data)
            if preds.max().item() - (1. / preds.size(0)) > 0.01 / preds.size(0):
                match_id = ids[preds.argmax().item()]
                query_track.id = match_id
                match = True
                
        write_lock.acquire()
        if not match:
            query_track.id = count_id.value
            count_id.value += 1
        result.append(query_track)
        write_lock.release()

def main(data, camera_dirs):
    manager  = mp.Manager()
    results  = dict()
    first_camera = camera_dirs[0].split("/")[-1]
    results[first_camera] = list(data[first_camera].values())
    count_id.value = len(results[first_camera]) + 1

    for camera_dir in camera_dirs[1:]:
        camera = camera_dir.split("/")[-1]
        query_tracks = data[camera]
        results[camera] = list()
        result = manager.list()
        processes = list()
        data_queue = mp.Queue()
        stop.value = False
        for gpu in GPUS:
            device = torch.device(DEVICE + ':' + str(gpu))
            p = mp.Process(target=match_track, args=(device, data_queue, result))
            p.start()
            processes.append(p)

        for obj_id in tqdm(query_tracks, desc=f"Processing Camera Dir {camera}"):
            gallery_tracks = list()
            query_track = query_tracks[obj_id]
            qid = query_track.id
            speed = query_track.speed()
            if speed == 0:
                continue
            gids = list()
            for c in results:
                for gallery_track in results[c]:
                    
                    g_x = gallery_track.gps_list[int(len(gallery_track)/2)][0] - gallery_track.gps_list[0][0]
                    g_y = gallery_track.gps_list[int(len(gallery_track)/2)][1] - gallery_track.gps_list[0][1]
                    q_x = query_track.gps_list[int(len(query_track)/2)][0] - query_track.gps_list[0][0]
                    q_y = query_track.gps_list[int(len(query_track)/2)][1] - query_track.gps_list[0][1]
                    vec1 = [g_x, g_y]
                    vec2 = [q_x, q_y]
                    direction = 1 - cosine(vec1, vec2)
                    dis_ts = abs(gallery_track.ts_list[0] - query_track.ts_list[0])
                    if np.isnan(direction):
                        continue
                    expected_time = getdistance(query_track.gps_list[0], gallery_track.gps_list[0]) / speed
                    
                    if (abs(dis_ts - expected_time) > 30) or (direction < 0.5) or (gallery_track.id in gids):
                        continue

                    gids.append(gallery_track.id)
                    gallery_tracks.append(gallery_track)

            if len(gallery_tracks) == 0:
                write_lock.acquire()
                query_track.id = count_id.value
                count_id.value += 1
                result.append(query_track)
                write_lock.release()
            else:
                while data_queue.qsize() >= 3:
                    pass
                read_lock.acquire()
                data_queue.put([query_track, gallery_tracks])
                read_lock.release()
        
        data_queue.close()
        data_queue.join_thread()
        stop.value = True
        for p in processes:
            p.join()
            
        results[camera] = result

    write_results(results, camera_dirs)
    
if __name__ == "__main__":
    data, camera_dirs = prepare_data()

    main(data, camera_dirs)