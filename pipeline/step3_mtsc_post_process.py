import os, cv2
import multiprocessing as mp
import numpy as np

from tqdm          import tqdm
from utils         import init_path, check_setting
from utils.objects import Track
from utils.utils   import getdistance, compute_iou

init_path()

from config import cfg

check_setting(cfg)

INPUT_DIR   = cfg.PATH.INPUT_PATH
NUM_WORKERS = mp.cpu_count()
write_lock  = mp.Lock()

SHORT_TRACK_TH = 2
IOU_TH = 0.1
STAY_TIME_TH = 20 # 30: 12.63, 20:12.80
MOVE_DIS_TH = 0.001

def halfway_appear(track, roi):
    side_th = 100
    h, w, _ = roi.shape
    bx = track.box_list[0]
    c_x, c_y = [int(float(bx[0])) + int(float(bx[2])/2), int(float(bx[1])) + int(float(bx[3])/2)]
    con1 = bx[0] > side_th
    con2 = bx[1] > side_th
    con3 = bx[0] + bx[2] < w - side_th
    con4 = bx[0] + bx[2] < w - side_th
    try:
        con5 = roi[c_y][c_x][0] > 128
    except:
        return False
    
    if con1 and con2 and con3 and con4 and con5:
        return True
    else:
        return False

def halfway_lost(track, roi):
    side_th = 100
    h, w, _ = roi.shape
    bx = track.box_list[-1]
    c_x, c_y = [int(float(bx[0])) + int(float(bx[2])/2), int(float(bx[1])) + int(float(bx[3])/2)]
    con1 = bx[0] > side_th
    con2 = bx[1] > side_th
    con3 = bx[0] + bx[2] < w - side_th
    con4 = bx[1] + bx[3] < h - side_th
    try:
        con5 = roi[c_y][c_x][0] > 128
    except:
        return False

    if con1 and con2 and con3 and con4 and con5:
        return True
    else:
        return False

def sort_tracks(tracks):
    sorted_tracks = dict()
    for track_id in tracks:
        track = tracks[track_id]
        track.sort()
        sorted_tracks[track_id] = track

    return sorted_tracks

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

def preprocess_roi(roi):
    h, w, _ = roi.shape
    width_erode = int(w * 0.1)
    height_erode = int(h * 0.1)
    left = roi[:, 0:width_erode, :]
    right = roi[:, w-width_erode:w, :]
    top = roi[0:height_erode, :, :]
    bottom = roi[h-height_erode:h, :, :]

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
        files.append(os.path.join(camera_dir, "all_features.txt"))
        
    pool = mp.Pool(NUM_WORKERS)

    for d in tqdm(pool.imap_unordered(read_features_file, files), total=len(files), desc="Loading Data"):
        data.update(d)

    pool.close()
    pool.join()

    return data, camera_dirs

def remove_short_track(tracks, threshold):
    delete_ids = list()
    for track_id in tracks:
        track = tracks[track_id]
        if len(track) < threshold:
            delete_ids.append(track_id)
    for track_id in delete_ids:
        tracks.pop(track_id)
    return tracks

def remove_overlapped_box(tracks, threshold):
    frame_dict = dict()
    res_tracks = dict()
    for track in tracks.values():
        for i in range(len(track)):
            fid = track.frame_list[i]
            if fid not in frame_dict:
                frame_dict[fid] = list()
            box = {
                "id": track.id,
                "gps": track.gps_list[i],
                "feature": track.feature_list[i],
                "box": track.box_list[i],
                "ts": track.ts_list[i] 
            }
            frame_dict[fid].append(box)

    for fid in frame_dict:
        boxes = frame_dict[fid]
        for cur_bx in boxes:
            box1 = cur_bx["box"]
            keep = True
            for cpr_bx in boxes:
                box2 = cpr_bx["box"]
                iou = compute_iou(box1, box2)
                cur_bottom = box1[1] + box1[3]
                cpr_bottom = box2[1] + box2[3]
                if iou > threshold and cur_bottom < cpr_bottom:
                    keep = False
                    break
            if keep:
                box = cur_bx["box"]
                track_id = cur_bx["id"]
                ts = cur_bx["ts"]
                fts = cur_bx["feature"]
                gps = cur_bx["gps"]
                
                if track_id not in res_tracks:
                    res_tracks[track_id] = Track()
                
                res_tracks[track_id].id = track_id
                res_tracks[track_id].frame_list.append(fid)
                res_tracks[track_id].box_list.append(box)
                res_tracks[track_id].ts_list.append(ts)
                res_tracks[track_id].gps_list.append(gps)
                res_tracks[track_id].feature_list.append(fts)
                
    return res_tracks

def remove_edge_box(tracks, roi):
    side_th = 30
    h, w, _ = roi.shape
    
    for track_id in tracks:
        boxes = tracks[track_id].box_list
        l = len(boxes)
        start = 0
        for i in range(0, l):
            bx = boxes[i]
            con1 = bx[0] > side_th
            con2 = bx[1] > side_th
            con3 = bx[0] + bx[2] < w - side_th
            con4 = bx[1] + bx[3] < h - side_th

            if con1 and con2 and con3 and con4:
                break
            else:
                start = i

        end = l-1
        for i in range(l-1, -1, -1):
            bx = boxes[i]
            con1 = bx[0] > side_th
            con2 = bx[1] > side_th
            con3 = bx[0] + bx[2] < w - side_th
            con4 = bx[1] + bx[3] < h - side_th

            if con1 and con2 and con3 and con4:
                break
            else:
                end = i
        end += 1

        if start >= end:
            tracks[track_id].box_list = boxes[0:1]
            tracks[track_id].feature_list = tracks[track_id].feature_list[0:1]
            tracks[track_id].frame_list = tracks[track_id].frame_list[0:1]
            tracks[track_id].gps_list = tracks[track_id].gps_list[0:1]
            tracks[track_id].ts_list = tracks[track_id].ts_list[0:1]
        else:
            tracks[track_id].box_list = boxes[start:end]
            tracks[track_id].feature_list = tracks[track_id].feature_list[start:end]
            tracks[track_id].frame_list = tracks[track_id].frame_list[start:end]
            tracks[track_id].gps_list = tracks[track_id].gps_list[start:end]
            tracks[track_id].ts_list = tracks[track_id].ts_list[start:end]

    return tracks

def remove_slow_tracks(tracks, threshold):
    delete_ids = list()
    for track_id in tracks:
        track = tracks[track_id]
        stay_time = track.ts_list[-1] - track.ts_list[0]
        if stay_time > threshold:
            delete_ids.append(track_id)

    for id in delete_ids:
        tracks.pop(id)
    
    return tracks

def connect_lost_tracks(tracks, roi):
    halfway_list = list()
    for track_id in tracks:
        track = tracks[track_id]
        if halfway_appear(track, roi) or halfway_lost(track, roi):
            halfway_list.append(track)

    halfway_list = sorted(halfway_list, key=lambda tk: tk.frame_list[0])
    delete_ids = list()
    for lost_tk in halfway_list:
        if lost_tk.id in delete_ids:
            continue
        for apr_tk in halfway_list:
            if apr_tk.id in delete_ids:
                continue
            if lost_tk.frame_list[-1] < apr_tk.frame_list[0]:
                dis = calu_track_distance(lost_tk, apr_tk)
                frame_dif = apr_tk.frame_list[0] - lost_tk.frame_list[-1]
                if frame_dif < 5:
                    th = 22
                else:
                    th = 8

                if dis < th:
                    for i in range(len(apr_tk)):
                        lost_tk.frame_list.append(apr_tk.frame_list[i])
                        lost_tk.feature_list.append(apr_tk.feature_list[i])
                        lost_tk.box_list.append(apr_tk.box_list[i])
                        lost_tk.gps_list.append(apr_tk.gps_list[i])
                        lost_tk.ts_list.append(apr_tk.ts_list[i])
                    delete_ids.append(apr_tk.id)
    
    for id in delete_ids:
        tracks.pop(id)
    
    return tracks

def remove_no_moving_tracks(tracks, threshold):
    delete_ids = list()
    for track_id in tracks:
        track = tracks[track_id]
        box1 = track.box_list[0]
        box2 = track.box_list[-1]
        iou = compute_iou(box1, box2)
        if iou > threshold:
            delete_ids.append(track_id)
        # pt1 = track.gps_list[0]
        # pt2 = track.gps_list[-1]
        # move_dis = getdistance(pt1, pt2)
        # if move_dis < threshold:
        #     delete_ids.append(track_id)

    for id in delete_ids:
        tracks.pop(id)
    
    return tracks

def main(_input):
    tracks, camera_dir = _input
    roi_path = os.path.join(camera_dir, 'roi.jpg')
    roi_src = cv2.imread(roi_path)
    roi = preprocess_roi(roi_src)
    tracks = remove_short_track(tracks, SHORT_TRACK_TH)
    tracks = connect_lost_tracks(tracks, roi)
    tracks = remove_edge_box(tracks, roi)
    tracks = remove_short_track(tracks, SHORT_TRACK_TH)
    tracks = remove_slow_tracks(tracks, STAY_TIME_TH)
    tracks = remove_no_moving_tracks(tracks, IOU_TH)
    tracks = remove_overlapped_box(tracks, IOU_TH) # +1 IDF1
    result_file_path = os.path.join(camera_dir, "all_features_post.txt")
    with open(result_file_path, "w") as f:
        for track in tracks.values():
            obj_id_str = str(track.id)
            for i in range(len(track)):
                frame_id_str = str(track.frame_list[i])
                ts_str = str(track.ts_list[i])
                gps_str = ",".join(list(map(str, track.gps_list[i])))
                box_str = ",".join(list(map(str, track.box_list[i])))
                feature_str = ",".join(list(map(str, track.feature_list[i])))
                line = frame_id_str + ',' + obj_id_str + ',' + box_str + \
                       ',' + ts_str + ',' + gps_str + ',' + feature_str + '\n'
                f.write(line)
    
if __name__ == "__main__":
    data, camera_dirs = prepare_data()
    pool = mp.Pool(NUM_WORKERS)
    print (f"Create {NUM_WORKERS} processes.")
    _input = list()
    for camera_dir in camera_dirs:
        camera_id = camera_dir.split('/')[-1]
        tracks = data[camera_id]
        tracks = sort_tracks(tracks)
        _input.append([tracks, camera_dir])
    del data
    for _ in tqdm(pool.imap_unordered(main, _input), total=len(_input), desc=f"Post Processing"):
        pass
    
    pool.close()
    pool.join()
    