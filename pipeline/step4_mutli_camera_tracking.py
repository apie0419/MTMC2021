import os, re, torch, cv2, math, argparse
import numpy as np
import torch.multiprocessing as mp

from tqdm          import tqdm
from utils         import init_path, check_setting
from utils.mct     import build_model
from utils.utils   import getdistance, cosine
from utils.objects import Track, GroupNode

init_path()

from config import cfg

check_setting(cfg)

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weight", type=str, default=cfg.MCT.WEIGHT)
parser.add_argument("-s", "--sim_th", type=float, default=cfg.MCT.SIM_TH)
parser.parse_args()
args = parser.parse_args()

cfg.MCT.WEIGHT = args.weight

ROOT_DIR    = cfg.PATH.ROOT_PATH
INPUT_DIR   = cfg.PATH.INPUT_PATH
NUM_WORKERS = mp.cpu_count()
DEVICE      = cfg.DEVICE.TYPE
GPUS        = cfg.DEVICE.GPUS
device      = torch.device(f"{DEVICE}:{GPUS[0]}")
METHOD      = cfg.MCT.METHOD
METRIC      = cfg.MCT.METRIC
SIM_TH      = args.sim_th
CIR_TH      = cfg.MCT.CIR_TH
RW          = cfg.MCT.RW

write_lock  = mp.Lock()

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

def get_feature_dict(data, camera_dirs):
    feature_dict = dict()
    for camera_dir in camera_dirs:
        camera = camera_dir.split("/")[-1]
        feature_dict[camera] = dict()
        tracks = data[camera]
        for obj_id in tracks:
            track = tracks[obj_id]
            feat = torch.tensor(track.feature_list)
            mean = feat.mean(dim=0)
            std = feat.std(dim=0, unbiased=False)
            total_feat = torch.cat((mean, std))
            feature_dict[camera][obj_id] = total_feat

    return feature_dict

def match_track_by_cosine(model, query_ft, gallery_fts):
    all_fts = [query_ft] + gallery_fts
    affinity_list = list()

    for qft in all_fts:
        fts_tensor = torch.stack(all_fts)
        aff = cosine(qft, fts_tensor.T)
        aff = torch.tensor(aff)
        affinity_list.append(aff)
    
    A = torch.stack(affinity_list)
    A = torch.nan_to_num(A)
    A = A.clamp(0, 1)
    with torch.no_grad():
        A = A.to(device)
        if RW and A.size(0) > 2:
            preds, g_preds = model.random_walk(A)
            preds = (preds - preds.mean())
            preds = preds * 10
            preds = torch.sigmoid(preds)
        else:
            preds = A[0][1:]
    
        sort_preds = torch.sort(preds, descending=True)
        std = preds.std()
        mean = preds.mean()
        
        match_idx = sort_preds.indices[sort_preds.values > SIM_TH]
        match_idx = match_idx.cpu().numpy().tolist()

    return match_idx

def match_track2(model, query_ft, gallery_fts):
    all_fts = [query_ft] + gallery_fts
    affinity_list = list()
    with torch.no_grad():
        for i, qft in enumerate(all_fts):
            aff = list()
            for j, gft in enumerate(all_fts):
                if i == j:
                    aff.append(0.)
                else:
                    fts_tensor = torch.stack([qft, gft])
                    fts_tensor = fts_tensor.to(device)
                    affinities = model(fts_tensor)
                    aff.append(affinities.cpu().numpy()[0][1])
            aff = torch.tensor(aff)
            affinity_list.append(aff)
        A = torch.stack(affinity_list)
        A = A.to(device)
        if RW and A.size(0) > 2:
            preds, g_preds = model.random_walk(A)
            preds = (preds - preds.mean())
            preds = preds * 10
            preds = torch.sigmoid(preds)
        else:
            preds = A[0][1:]
        
        sort_preds = torch.sort(preds, descending=True)
        std = preds.std()
        mean = preds.mean()
        match_idx = sort_preds.indices[sort_preds.values > SIM_TH]
        match_idx = match_idx.cpu().numpy().tolist()
    
    return match_idx

def match_track(model, query_ft, gallery_fts):
    
    tracklets = [query_ft]
    tracklets.extend(gallery_fts)
    data = torch.stack(tracklets)
    with torch.no_grad():
        data = data.to(device)
        
        A = model(data)
        if RW and A.size(0) > 2:
            preds, g_preds = model.random_walk(A)
            preds = (preds - preds.mean())
            preds = preds * 10
            preds = torch.sigmoid(preds)
        else:
            preds = A[0][1:]
        
        sort_preds = torch.sort(preds, descending=True)
        std = preds.std()
        mean = preds.mean()
        match_idx = sort_preds.indices[sort_preds.values > SIM_TH]
        match_idx = match_idx.cpu().numpy().tolist()
    
    return match_idx

def CIR(A, B):
    inter_num = 0.
    if len(B) == 0:
        return 0

    for camera in A:
        if camera in B:
            if B[camera] == A[camera]:
                inter_num += 1

    return inter_num / len(B)

def grouping_matches(match_dict, scene_dict):
    for qc in tqdm(match_dict, desc=f"Grouping Matches"):
        qs = scene_dict[qc]
        for qid in match_dict[qc]:
            nodeA = match_dict[qc][qid]
            for gc in match_dict:
                gs = scene_dict[gc]
                if gc==qc:
                    continue
                for gid in match_dict[gc]:
                    nodeB = match_dict[gc][gid]
                    A = nodeA.match_ids.copy()
                    B = nodeB.match_ids.copy()
                    lenA = len(A)
                    lenB = len(B)
                    if lenA < lenB:
                        continue
                    score = 0
                    if (gc in A and gid == A[gc]) and (qc in B and qid == B[qc]):
                        score = 1
                    else:
                        if gc in A:
                            A.pop(gc)
                        if qc in B:
                            B.pop(qc)
                        score = CIR(A, B)

                    if lenA == lenB:
                        normal = True
                        if nodeA.parent != None:
                            parentnode = nodeA.parent
                            while parentnode != None:
                                # print (parentnode.id)
                                if parentnode.id == nodeB.id:
                                    normal = False
                                    break
                                parentnode = parentnode.parent
                        if normal:
                            if score > nodeB.max_intersection:
                                nodeB.parent = nodeA
                                nodeB.max_intersection = score
                                match_dict[gc][gid] = nodeB
                        else:
                            if score > nodeA.max_intersection:
                                nodeA.parent = nodeB
                                nodeA.max_intersection = score
                                match_dict[qc][qid] = nodeA
                    else:
                        if score > nodeB.max_intersection:
                            nodeB.parent = nodeA
                            nodeB.max_intersection = score
                            match_dict[gc][gid] = nodeB

    id_counter = dict()
    for camera in tqdm(match_dict, desc="Setting ID"):
        for id in match_dict[camera]:
            if match_dict[camera][id].parent != None:
                match_dict[camera][id].set_parent_id()
            _id = match_dict[camera][id].id
            if _id not in id_counter:
                id_counter[_id] = 1
            else:
                id_counter[_id] += 1
    delete_ids = list()
    
    for camera in match_dict:
        for id in match_dict[camera]:
            _id = match_dict[camera][id].id
            if id_counter[_id] == 1:
                delete_ids.append((camera, id))

    for camera, id in delete_ids:
        match_dict[camera].pop(id)

    return match_dict

def main(data, camera_dirs):
    results  = dict()
    camera_dirs.sort()
    scene_dict = get_scene_camera_dict(camera_dirs)
    feature_dict = get_feature_dict(data, camera_dirs)
    match_dict = dict()
    count_id = 1
    model = build_model(cfg, device)
    model.eval()
    for q_camera_dir in camera_dirs:
        q_camera = q_camera_dir.split("/")[-1]
        query_scene = scene_dict[q_camera]
        query_tracks = data[q_camera]
        match_dict[q_camera] = dict()
        results[q_camera] = list()
        for qid in tqdm(query_tracks, desc=f"Processing Camera Dir {q_camera}"):
            gallery_fts = list()
            query_track = query_tracks[qid]
            speed = query_track.speed()
            query_ft = feature_dict[q_camera][qid]
            gallery_fts = list()
            idx_camera_dict = dict()
            gids = list()
            if METHOD == "CIR":
                g_camera_dirs = camera_dirs
            elif METHOD == "top1":
                g_camera_dirs = list(match_dict.keys())
            
            for g_camera_dir in g_camera_dirs:
                g_camera = g_camera_dir.split("/")[-1]
                gallery_scene = scene_dict[g_camera]
                if gallery_scene != query_scene or g_camera == q_camera:
                    continue
                for gid in data[g_camera]:
                    gallery_track = data[g_camera][gid]
                    
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
                    if abs(expected_time - dis_ts) > 30 or (direction < -0.5):
                        continue
                    
                    gallery_fts.append(feature_dict[g_camera][gid])
                    gids.append(gid)
                    idx_camera_dict[len(gallery_fts)-1] = g_camera
            if METHOD == "CIR":
                if len(gallery_fts) > 0:
                    if METRIC == "model":
                        match_idx = match_track(model, query_ft, gallery_fts)
                    elif METRIC == "cosine":
                        match_idx = match_track_by_cosine(model, query_ft, gallery_fts)
                    if len(match_idx) == 0:
                        continue
                    match_cameras = list()
                    match_ids = dict()
                    for idx in match_idx:
                        c = idx_camera_dict[idx]
                        if c in match_cameras:
                            match_idx.remove(idx)
                        else:
                            match_cameras.append(c)
                            gid = gids[idx]
                            match_ids[c] = gid
                    match_dict[q_camera][qid] = GroupNode(match_ids, count_id, CIR_TH)
                    count_id += 1

            elif METHOD == "top1":
                match = False
                if len(gallery_fts) > 0:
                    if METRIC == "model":
                        match_idx = match_track(model, query_ft, gallery_fts)
                    elif METRIC == "cosine":
                        match_idx = match_track_by_cosine(model, query_ft, gallery_fts)
                    if len(match_idx) > 0:
                        idx = match_idx[0]
                        g_camera = idx_camera_dict[idx]
                        gid = gids[idx]
                        match_id = data[g_camera][gid].id
                        match = True
                if match:
                    query_track.id = match_id
                    data[q_camera][qid] = query_track
                else:
                    match_id = count_id
                    query_track.id = match_id
                    data[q_camera][qid] = query_track
                    count_id += 1
                results[q_camera].append(query_track)

    if METHOD == "CIR":
        group_results = grouping_matches(match_dict, scene_dict)
        for camera in group_results:
            for id in group_results[camera]:
                node = group_results[camera][id]
                track = data[camera][id]
                track.id = node.id
                results[camera].append(track)

    write_results(results, camera_dirs)
    
if __name__ == "__main__":
    data, camera_dirs = prepare_data()
    main(data, camera_dirs)
    