import numpy as np
import os, time, torch, sys

from tqdm       import tqdm
from PIL        import Image
from utils.sct  import build_model
from utils.reid import build_transform
from utils                                  import init_path

init_path()

from config                                 import cfg
from sct_model.clusters.init_cluster        import init_clustering
from sct_model.clusters.optimal_cluster     import get_optimal_spfa
from sct_model.clusters.utils.cluster_utils import cluster_dict_processing
from sct_model.TNT.utils.merge_det          import merge_det

INPUT_DIR     = cfg.PATH.INPUT_PATH
DEVICE        = cfg.DEVICE.TYPE

if DEVICE == "cuda":
    torch.cuda.set_device(cfg.DEVICE.GPU)

def read_feature_file(file_path, transform):
    
    root_path = "/".join(file_path.split("/")[:-1])
    f = open(file_path, "r")
    det_result = dict()
    crop_imgs = dict()
    
    for line in tqdm(f.readlines(), desc="Reading feature files"):
        feature_list = line.strip("\n").split(",")
        img_name = feature_list[0]
        frame_id = int(feature_list[1])
        if frame_id not in det_result:
            det_result[frame_id] = list()
            crop_imgs[frame_id] = list()
        tmp_result = list()
        box = feature_list[3:7]
        gps = feature_list[9:11]
        emb = feature_list[11:]
        id = len(det_result[frame_id])
        tmp_result.extend(emb)
        tmp_result.extend(box)
        tmp_result.extend(gps)
        tmp_result.append(id)
        tmp_result = np.array(tmp_result).astype(np.float)
        img_path = os.path.join(root_path, "cropped_images", img_name)
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        det_result[frame_id].append(tmp_result)
        crop_imgs[frame_id].append(img.numpy())
    
    for key in det_result.keys():
        det_result[key] = np.array(det_result[key])

    return det_result, crop_imgs

def main(working_path, model, transform):
    sct_file = os.path.join(working_path, "sct_results.txt")
    if os.path.exists(sct_file):
        os.remove(sct_file)
    f_track = open(sct_file, "a+")
    feature_file = os.path.join(working_path, "all_features.txt")
    det_result, crop_im = read_feature_file(feature_file, transform)

    coarse_track_dict = merge_det(det_result, crop_im)

    cluster_dict, tracklet_time_range, coarse_tracklet_connects, tracklet_cost_dict = init_clustering(model, coarse_track_dict)

    min_cost, cluster_list = get_optimal_spfa(coarse_tracklet_connects, tracklet_cost_dict)

    # post processing
    for cluster in cluster_list:
        cluster_id = min(cluster)
        for track_id in cluster:
            if track_id == cluster_id:
                continue
            if track_id in cluster_dict.keys():
                cluster_dict.pop(track_id)
            if cluster_id not in cluster_dict.keys():
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(track_id)

    # generate cluster feat dict, cluster_id: np.array((frame_len, emb_size+4+1)), xmin, ymin, w, h
    cluster_feat_dict, cluster_frame_range = cluster_dict_processing(coarse_track_dict, tracklet_time_range, cluster_dict)
    
    frame_len = len(cluster_feat_dict[0])
    
    for cluster_id in cluster_feat_dict.keys():
        loc = cluster_feat_dict[cluster_id][:, -5:-1] # (frame_len, 4)
        # xmin, ymin, xmax, ymax
        loc[:, 2] += loc[:, 0]
        loc[:, 3] += loc[:, 1]
        
        min_t = int(cluster_frame_range[cluster_id][0])
        max_t = int(cluster_frame_range[cluster_id][1])
        if max_t - min_t + 1 > 5:
            print(cluster_id, cluster_frame_range[cluster_id], class_dict[int(label)])
            f_track.writelines(f'{int(cluster_frame_range[cluster_id][0])}, {int(cluster_frame_range[cluster_id][1])}, {class_dict[int(label)]}\n')

    f_track.close()

if __name__ == '__main__':
    transform = build_transform(cfg)
    model = build_model(cfg)
    model.to(DEVICE)
    scene_dirs = []
    scene_fds = os.listdir(INPUT_DIR)
    for scene_fd in scene_fds:
        if scene_fd.startswith("S0"):
            scene_dirs.append(os.path.join(INPUT_DIR, scene_fd))

    for scene_dir in scene_dirs:
        working_dirs = []
        camera_dirs = os.listdir(scene_dir)
        for camera_dir in camera_dirs:
            if camera_dir.startswith('c0'):
                working_dirs.append(os.path.join(scene_dir, camera_dir))
        for working_dir in working_dirs:
            main(working_dir, model, transform)







    
    


    
    



    


    






