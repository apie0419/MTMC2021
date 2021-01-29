import os, re, torch, cv2, math
import numpy as np
import multiprocessing as mp

from tqdm      import tqdm
from utils     import init_path, check_setting
from utils.mct import build_model

init_path()

from config import cfg

check_setting(cfg)

ROOT_DIR    = cfg.PATH.ROOT_PATH
INPUT_DIR   = cfg.PATH.INPUT_PATH
NUM_WORKERS = mp.cpu_count()
DEVICE      = cfg.DEVICE.TYPE
GPUS        = cfg.DEVICE.GPUS

device      = torch.device(DEVICE + ':' + str(GPUS[0]))
write_lock  = mp.Lock()

class Track(object):

    def __init__(self):
        self.id = None
        self.gps_list = list()
        self.feature_list = list()
        self.frame_list = list()
        self.box_list = list()
    
    def __len__(self):
        return len(self.frame_list)

def getdistance(pt1, pt2):
    EARTH_RADIUS = 6378.137
    lat1, lon1 = pt1[0], pt1[1]
    lat2, lon2 = pt2[0], pt2[1]
    radlat1 = lat1 * math.pi / 180
    radlat2 = lat2 * math.pi / 180
    lat_dis = radlat1 - radlat2
    lon_dis = (lon1 * math.pi - lon2 * math.pi) / 180
    distance = 2 * math.asin(math.sqrt((math.sin(lat_dis/2) ** 2) + math.cos(radlat1) * math.cos(radlat2) * (math.sin(lon_dis/2) ** 2)))
    distance *= EARTH_RADIUS
    distance = round(distance * 10000) / 10000
    return distance

def get_timestamp_dict():
    ts_dict = dict()
    for filename in os.listdir(os.path.join(ROOT_DIR, "cam_timestamp")):
        with open(os.path.join(ROOT_DIR, "cam_timestamp", filename), "r") as f:
            lines = f.readlines()
            temp = dict()
            for line in lines:
                split_line = line.strip("\n").split(" ")
                temp[split_line[0]] = float(split_line[1])
            _max = np.array(list(temp.values())).max()
            for camid, ts in temp.items():
                ts_dict[camid] = ts * -1 + _max

    return ts_dict

def get_fps_dict():
    fps_dict = dict()
    for scene_dir in os.listdir(INPUT_DIR):
        if scene_dir.startswith("S0"):
            for camera_dir in os.listdir(os.path.join(INPUT_DIR, scene_dir)):
                if camera_dir.startswith("c0"):
                    video_path = os.path.join(INPUT_DIR, scene_dir, camera_dir, "vdo.avi")
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    fps_dict[camera_dir] = float(fps)
    return fps_dict

def cosine(vec1, vec2):
    
    num = float(np.matmul(vec1, vec2))
    s = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if s ==0:
        result = 0.0
    else:
        result = num/s
    return result

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
        features = np.array(list(map(float, l[8:])), np.float32)
        
        write_lock.acquire()
        if id not in data[camera_dir]:
            data[camera_dir][id] = Track()
        track = data[camera_dir][id]
        track.gps_list.append(gps)
        track.feature_list.append(features)
        track.frame_list.append(frame_id)
        track.box_list.append(box)
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
        
def match_track(model, query_track, gallery_tracks):
    qid = query_track.id
    query_track = torch.tensor(query_track.feature_list)
    mean = query_track.mean(dim=0)
    std = query_track.std(dim=0, unbiased=False)
    query = torch.cat((mean, std))
    tracklets = [query]
    ids = list()
    for track in gallery_tracks:
        gallery_track = torch.tensor(track.feature_list)
        mean = gallery_track.mean(dim=0)
        std  = gallery_track.std(dim=0, unbiased=False)
        gallery = torch.cat((mean, std))
        tracklets.append(gallery)
        ids.append(track.id)
        
    data = torch.stack(tracklets)
    with torch.no_grad():
        data = data.to(device)
        preds = model(data)
        if preds.max().item() - (1. / preds.size(0)) > 0.1 / preds.size(0):
            match_id = ids[preds.argmax().item()]
        else:
            match_id = qid

    return match_id

def main(data, camera_dirs):
    results = dict()
    ts_dict = get_timestamp_dict()
    fps_dict = get_fps_dict()
    first_camera = camera_dirs[0].split("/")[-1]
    results[first_camera] = list(data[first_camera].values())
    model = build_model(cfg, device)
    model.eval()
    for camera_dir in camera_dirs[1:]:
        camera = camera_dir.split("/")[-1]
        query_tracks = data[camera]
        results[camera] = list()
        qt_ts = ts_dict[camera]
        qt_ts_per_frame = 1/fps_dict[camera]
        
        for obj_id in tqdm(query_tracks, desc=f"Processing Camera Dir {camera}"):
            gallery_tracks = list()
            query_track = query_tracks[obj_id]
            qid = query_track.id
            first_gps = query_track.gps_list[0]
            last_gps = query_track.gps_list[-1]
            dis = getdistance(first_gps, last_gps)
            ts = abs(query_track.frame_list[0] - query_track.frame_list[-1]) * qt_ts_per_frame
            if dis == 0 or ts == 0:
                results[camera].append(query_track)
                continue
            speed = dis / ts
            gids = list()
            for c in results:
                gt_ts = ts_dict[c]
                gt_ts_per_frame = 1/fps_dict[c]
                for gallery_track in results[c]:
                    
                    g_x = gallery_track.gps_list[int(len(gallery_track)/2)][0] - gallery_track.gps_list[0][0]
                    g_y = gallery_track.gps_list[int(len(gallery_track)/2)][1] - gallery_track.gps_list[0][1]
                    q_x = query_track.gps_list[int(len(query_track)/2)][0] - query_track.gps_list[0][0]
                    q_y = query_track.gps_list[int(len(query_track)/2)][1] - query_track.gps_list[0][1]
                    vec1 = [g_x, g_y]
                    vec2 = [q_x, q_y]
                    direction = 1 - cosine(vec1, vec2)
                    dis_ts = abs((gallery_track.frame_list[0] * gt_ts_per_frame + gt_ts) - (query_track.frame_list[0] * qt_ts_per_frame + qt_ts))
                    if np.isnan(direction):
                        continue
                    expected_time = getdistance(query_track.gps_list[0], gallery_track.gps_list[0]) / speed
                    
                    if (direction < 0) or (gallery_track.id in gids):
                        continue

                    gids.append(gallery_track.id)
                    gallery_tracks.append(gallery_track)
            if len(gallery_tracks) == 0:
                results[camera].append(query_track)
            else:
                match_id = match_track(model, query_track, gallery_tracks)
                query_track.id = match_id
                results[camera].append(query_track)

    write_results(results, camera_dirs)
    

if __name__ == "__main__":
    data, camera_dirs = prepare_data()

    main(data, camera_dirs)