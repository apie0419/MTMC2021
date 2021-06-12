import os, torch, random, numpy as np

from tqdm        import tqdm
from utils       import init_path
from config      import cfg

init_path()

path = cfg.PATH.VALID_PATH
tracklets_file = os.path.join(path, "gt_features.txt")
easy_output_file = os.path.join(path, "mtmc_easy_binary_multicam.txt")
hard_output_file = os.path.join(path, "mtmc_hard_binary_multicam.txt")

def read_feature_file(filename):
    data_dict = dict()
    frame_dict = dict()
    with open(filename, 'r') as f:
        for line in tqdm(f.readlines(), desc="Reading Tracklets File"):
            words = line.strip("\n").split(',')
            camera_id = words[0]
            frame_id = int(words[1])
            det_id = words[2]
            features = list(map(float, words[3:]))
            if camera_id not in data_dict:
                data_dict[camera_id] = dict()
                frame_dict[camera_id] = dict()
            if det_id not in data_dict[camera_id]:
                data_dict[camera_id][det_id] = list()
                frame_dict[camera_id][det_id] = list()

            data_dict[camera_id][det_id].append(features)
            frame_dict[camera_id][det_id].append(frame_id)

    return data_dict, frame_dict

def get_id_dict(data_dict):
    id_dict = dict()
    for camera_id in data_dict:
        cam_data = data_dict[camera_id]
        for _id in cam_data:
            if _id not in id_dict:
                id_dict[_id] = [camera_id]
            else:
                id_dict[_id].append(camera_id)
    return id_dict

def write_results(res_dict, filename):
    with open(filename, "w+") as f:
        for q_camera in res_dict:
            for q_id in res_dict[q_camera]:
                for gallery_tracks in res_dict[q_camera][q_id]:
                    gallery_str = list()
                    for gt in gallery_tracks:
                        gallery_str.append("/".join(gt))
                    gallery_str = ",".join(gallery_str)
                    line = q_camera + ' ' + q_id + ' ' + gallery_str + '\n'
                    f.write(line)

def main():
    data_dict, frame_dict = read_feature_file(tracklets_file)
    id_dict = get_id_dict(data_dict)
    hard_res_dict = dict()
    easy_res_dict = dict()
    all_cameras = list(data_dict.keys())
    for qid in tqdm(id_dict, desc="Preparing Data"):
        q_exist_cams = id_dict[qid]
        if len(q_exist_cams) < 3:
            continue
        
        for qcam in q_exist_cams:
            if qcam not in hard_res_dict:
                hard_res_dict[qcam] = dict()
            if qid not in hard_res_dict[qcam]:
                hard_res_dict[qcam][qid] = list()
            if qcam not in easy_res_dict:
                easy_res_dict[qcam] = dict()
            if qid not in easy_res_dict[qcam]:
                easy_res_dict[qcam][qid] = list()
            query_track = data_dict[qcam][qid]
            query_track = torch.tensor(query_track)
            mean = query_track.mean(dim=0)
            std = query_track.std(dim=0, unbiased=False)
            pos_sim = list()
            query = torch.cat((mean, std))
            hard_samples = list()
            easy_samples = list()
            pos_num = len(q_exist_cams) - 1
            ## Positive
            for p_gcam in q_exist_cams:
                if qcam != p_gcam:
                    gallery_track = torch.tensor(data_dict[p_gcam][qid])
                    mean = gallery_track.mean(dim=0)
                    std  = gallery_track.std(dim=0, unbiased=False)
                    gallery = torch.cat((mean, std))
                    num = float(torch.matmul(query, gallery))
                    s = torch.norm(query) * torch.norm(gallery)
                    if s == 0:
                        cos = 0.0
                    else:
                        cos = num/s
                    pos_sim.append(cos)

            ## Negative
            gid_list = list(id_dict.keys())
            random.shuffle(gid_list)
            for gid in gid_list:
                if qid == gid:
                    continue
                g_exist_cams = id_dict[gid]
                for gcam in g_exist_cams:
                    if gcam == qcam:
                        continue
                    gallery_track = torch.tensor(data_dict[gcam][gid])
                    mean = gallery_track.mean(dim=0)
                    std  = gallery_track.std(dim=0, unbiased=False)
                    gallery = torch.cat((mean, std))
                    num = float(torch.matmul(query, gallery))
                    s = torch.norm(query) * torch.norm(gallery)
                    if s == 0:
                        cos = 0.0
                    else:
                        cos = num/s

                    hard = False
                    for pos_cos in pos_sim:
                        if pos_cos <= cos:
                            hard = True
                            break
                    
                    if hard:
                        hard_samples.append((gcam, gid))
                    else:
                        easy_samples.append((gcam, gid))
                    if path == cfg.PATH.VALID_PATH:
                        if len(hard_samples) >= pos_num and len(easy_samples) >= pos_num:
                            break
                if path == cfg.PATH.VALID_PATH:
                    if len(hard_samples) >= pos_num and len(easy_samples) >= pos_num:
                        break
            if path == cfg.PATH.VALID_PATH:
                length = min(len(hard_samples), len(easy_samples), 1)
            else:
                length = min(len(hard_samples), len(easy_samples)) - pos_num

            for i in range(0, length, pos_num):
                e_samples = list()
                h_samples = list()
                for g_pcam in q_exist_cams:
                    if g_pcam != qcam:
                        e_samples.insert(0, (g_pcam, qid))
                        h_samples.insert(0, (g_pcam, qid))
                e_samples.extend(easy_samples[i:i+pos_num])
                h_samples.extend(hard_samples[i:i+pos_num])
                random.shuffle(e_samples)
                random.shuffle(h_samples)
                easy_res_dict[qcam][qid].append(e_samples)
                hard_res_dict[qcam][qid].append(h_samples)
                
    write_results(hard_res_dict, hard_output_file)
    write_results(easy_res_dict, easy_output_file)

if __name__ == "__main__":
    main()