import os, cv2
import multiprocessing as mp

from tqdm  import tqdm
from utils import init_path, check_setting

init_path()

from config import cfg

check_setting(cfg)

INPUT_DIR   = cfg.PATH.INPUT_PATH
NUM_WORKERS = mp.cpu_count()
write_lock  = mp.Lock()

class Track(object):

    def __init__(self):
        self.id = None
        self.frame_list = list()
        self.box_list = list()
        self.gps_list = list()
        self.feature_list = list()
        
    def __len__(self):
        return len(self.frame_list)

def halfway_appear(track, roi):
    side_th = 100
    h, w, _ = roi.shape
    bx = track.box_list[0]
    c_x, c_y = [int(float(bx[0])) + int(float(bx[2]))/2, int(float(bx[1])) + int(float(bx[3]))/2]
    if bx[0] > side_th and bx[1] > side_th and bx[0] + bx[2] < w - side_th and bx[1] + bx[3] < h - side_th and roi[c_y][c_x][0] > 128:
        return True
    else:
        return False

def halfway_lost(track, roi):
    side_th = 100
    h, w, _ = roi.shape
    bx = track.box_list[-1]
    c_x, c_y = [int(float(bx[0])) + int(float(bx[2]))/2, int(float(bx[1])) + int(float(bx[3]))/2]

    if bx[0] > side_th and bx[1] > side_th and bx[0] + bx[2] < w - side_th and bx[1] + bx[3] < h - side_th and roi[c_y][c_x][0] > 128:
        return True
    else:
        return False

def calu_track_distance(pre_tk, back_tk):
    lp = min(5, len(pre_tk)) * -1
    lb = min(5, len(back_tk))

    pre_seq = pre_tk.feature_list[lp:]
    back_seq = back_tk.feature_list[:lb]

    mindis = 999999
    for ft0 in pre_seq:
        for ft1 in back_seq:
            feature_dis_vec = ft1 - ft0
            curdis = np.dot(feature_dis_vec.T, feature_dis_vec)
            mindis = min(curdis, mindis)
            
    return mindis

def remove_edge_box(tracks, roi):
    side_th = 30
    h, w, _ = roi.shape
    for track_id in tracks:
        boxes = tracks[track_id].box_list
        l = len(track)
        start = 0
        for i in range(0, l):
            bx = boxes[i]
            if bx[0] > side_th and bx[1] > side_th and bx[0] + bx[2] < w - side_th and bx[1] + bx[3] < h - side_th:
                break
            else:
                start = i

        end = l-1
        for i in range(l-1, -1, -1):
            bx = boxes[i]
            if bx[0] > side_th and bx[1] > side_th and bx[0] + bx[2] < w - side_th and bx[1] + bx[3] < h - side_th:
                break
            else:
                end = i
        end += 1

        if start >= end:
            tracks[track_id].box_list = boxes[0:1]
            tracks[track_id].feature_list = tracks[track_id].feature_list[0:1]
            tracks[track_id].frame_list = tracks[track_id].frame_list[0:1]
            tracks[track_id].gps_list = tracks[track_id].gps_list[0:1]
        else:
            tracks[track_id].box_list = boxes[start:end]
            tracks[track_id].feature_list = tracks[track_id].feature_list[start:end]
            tracks[track_id].frame_list = tracks[track_id].frame_list[start:end]
            tracks[track_id].gps_list = tracks[track_id].gps_list[start:end]

    return tracks

def preprocess_roi(roi):
    h, w, _ = roi.shape
    width_erode = int(w * 0.1)
    height_erode = int(h * 0.1)
    left = roi[:, 0:width_erode, :]
    right = roi[:, w-width_erode:w, :]
    top = roi[0:height_erode, :, :]
    bottom = roi[h-height_erode:h, :, :]

    left = left*0
    right = right*0
    top = top*0
    bottom = bottom*0

    return roi

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
        files.append(os.path.join(camera_dir, f"mtsc/mtsc_{cfg.SCT}_{cfg.DETECTION}.txt"))
        
    pool = mp.Pool(NUM_WORKERS)

    for d in tqdm(pool.imap_unordered(read_features_file, files), total=len(files), desc="Loading Data"):
        data.update(d)

    pool.close()
    pool.join()

    return data, camera_dirs

def main():
    data, camera_dirs = prepare_data()
    for camera_dir in camera_dirs:
        camera_id = camera_dir.split('/')[-1]
        roi_path = os.path.join(camera_dir, 'roi.jpg')
        tracks = data[camera_id]
        halfway_list = list()
        roi_src = cv2.imread(roi_path)
        roi = preprocess_roi(roi_src)
        for track_id in tracks:
            track = tracks[track_id]
            
            if len(track) < 2:
                del data[camera_id][track_id]
                continue

            if halfway_appear(track, roi) or halfway_lost(track, roi):
                    halfway_list.append(track)
                else:
                    continue

        halfway_list = sorted(halfway_list, key=lambda tk: tk.frame_list[0])
        delete_list = list()
        for lost_tk in halfway_list:
            if lost_tk.id not in data[camera_id].keys():
                continue
            for apr_tk in halfway_list:
                if apr_tk.id not in data[camera_id].keys():
                    continue
                if lost_tk.frame_list[-1] < apr_tk.frame_list[0]:
                    dis = calu_track_distance(lost_tk, apr_tk)
                    time = apr_tk.frame_list[0] - lost_tk.frame_list[-1]
                    if time < 5:
                        th = 22
                    else:
                        th = 8

                    if dis < th:
                        for i in range(len(apr_tk)):
                            lost_tk.frame_list.append(apr_tk.frame_list[i])
                            lost_tk.feature_list.append(apr_tk.feature_list[i])
                            lost_tk.box_list.append(apr_tk.box_list[i])
                            lost_tk.gps_list.append(apr_tk.gps_list[i])
                        del data[camera_id][apr_tk.id]



        tracks = remove_edge_box(tracks, roi)
        data[camera_dir] = tracks

        for track_id in tracks:
            track = tracks[track_id]
            
            if len(track) < 2:
                del data[camera_id][track_id]

    